# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from torchvision.ops import box_iou
from torchvision.ops import nms

from mmdet.models import FocalLoss
from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import InstanceList
from .bbox_head import BBoxHead


def get_fg_score_pca_no_center(fix_concept_feat, query_feats, n_components=24, temp=0.1):
    """
    使用 PCA 子空间重构计算前景得分（不去中心化版本）

    参数:
        fix_concept_feat: [N_bg, 1024] - 纯净的背景参考特征
        query_feats:      [N_query, 1024] - 待测目标特征
        n_components:     int - PCA 保留的主成分数量
        temp:             float - 温度参数

    返回:
        fg_scores: [N_query, 1] - 归一化后的前景得分
    """

    import torch.nn.functional as F

    # 1. L2 归一化
    bg_feats = F.normalize(fix_concept_feat, p=2, dim=1)
    q_feats = F.normalize(query_feats, p=2, dim=1)

    # 2. 直接计算 PCA（center=False 表示不去中心化）
    with torch.cuda.amp.autocast(enabled=False):
        U, S, V = torch.pca_lowrank(bg_feats, q=n_components, center=False)

    # 3. 投影与重构（注意这里不需要去中心化）
    projected_coeffs = torch.mm(q_feats, V)
    reconstructed_feats = torch.mm(projected_coeffs, V.t())

    # 4. 计算重构误差
    diff = q_feats - reconstructed_feats
    recon_error = torch.norm(diff, dim=1, p=2)

    fg_scores = torch.sigmoid((recon_error - 0.5) / temp)
    #
    # return fg_scores  # [N, 1]
    # --- 7. 归一化得分 (Min-Max) ---
    # 将误差映射到 0-1
    # e_min = recon_error.min()
    # e_max = recon_error.max()
    #
    # fg_scores = (recon_error - e_min) / (e_max - e_min + 1e-8)

    return fg_scores  # [N, 1]


def filter_by_ref_keep_bboxes(
        keep_bboxes,
        keep_labels,
        ref_keep_bboxes,
        iou_threshold=0.5,
        min_size=10,
):
    if keep_bboxes.numel() == 0:
        return keep_bboxes, keep_labels

    bboxes = keep_bboxes[:, :4]  # [N, 4]
    scores = keep_bboxes[:, 4]  # [N]

    ref_bboxes = ref_keep_bboxes[:, :4]

    if ref_bboxes.numel() == 0:
        return keep_bboxes, keep_labels

    ious = box_iou(bboxes, ref_bboxes)  # [N, M]
    max_iou, _ = ious.max(dim=1)  # [N]

    keep_mask = max_iou <= iou_threshold

    # -------- Step 3: 过滤 --------
    bboxes = bboxes[keep_mask]
    scores = scores[keep_mask]
    labels = keep_labels[keep_mask]

    if bboxes.numel() == 0:
        return (
            keep_bboxes.new_zeros((0, 5)),
            keep_labels.new_zeros((0,), dtype=keep_labels.dtype),
        )

    # -------- 增加：过滤掉边长小于 10 的框 --------
    # 假设 bboxes 格式为 [x1, y1, x2, y2]
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]

    # 生成掩码：宽度和高度都必须大于等于 10
    keep_size_mask = (widths >= min_size) & (heights >= min_size)

    # 过滤数据
    bboxes = bboxes[keep_size_mask]
    scores = scores[keep_size_mask]
    labels = labels[keep_size_mask]

    # -------- Step 4: 按 score 排序 + 截断 --------
    order = scores.argsort(descending=True)

    bboxes = bboxes[order]
    scores = scores[order]
    labels = labels[order]

    keep_bboxes = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)

    return keep_bboxes, labels


def class_agnostic_nms(bboxes, scores, iou_threshold=0.5, score_threshold=0.05, max_num=100, score_weight=1.0):
    scores, labels = scores.max(dim=1)
    # Step 1: 先按 score 排序
    score_sorted_idx = scores.argsort(descending=True)
    bboxes = bboxes[score_sorted_idx]
    scores = scores[score_sorted_idx]
    labels = labels[score_sorted_idx]

    # 预筛选score
    keep_mask = scores > score_threshold
    filtered_bboxes = bboxes[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_labels = labels[keep_mask]

    keep = nms(filtered_bboxes, filtered_scores, iou_threshold)
    keep = keep[:max_num]
    keep_bboxes = filtered_bboxes[keep]
    keep_scores = filtered_scores[keep].unsqueeze(-1)
    keep_labels = filtered_labels[keep]
    keep_scores = keep_scores * score_weight
    keep_bboxes = torch.cat((keep_bboxes, keep_scores), dim=1)

    return keep_bboxes, keep_labels


def class_agnostic_nms_with_size(bboxes, scores, min_size=20, iou_threshold=0.5, score_threshold=0.05, max_num=100,
                                 score_weight=1.0):
    scores, labels = scores.max(dim=1)

    # Step 1: Sort by score
    score_sorted_idx = scores.argsort(descending=True)
    bboxes = bboxes[score_sorted_idx]
    scores = scores[score_sorted_idx]
    labels = labels[score_sorted_idx]

    # Step 2: Preliminary score filter
    keep_mask = scores > score_threshold
    filtered_bboxes = bboxes[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_labels = labels[keep_mask]

    # --- Added: Filter boxes with side length < 10 ---
    if filtered_bboxes.numel() > 0:
        ws = filtered_bboxes[:, 2] - filtered_bboxes[:, 0]
        hs = filtered_bboxes[:, 3] - filtered_bboxes[:, 1]
        # Keep only boxes where BOTH width and height >= 10
        size_mask = (ws >= min_size) & (hs >= min_size)

        filtered_bboxes = filtered_bboxes[size_mask]
        filtered_scores = filtered_scores[size_mask]
        filtered_labels = filtered_labels[size_mask]
    # -----------------------------------------------

    if filtered_bboxes.numel() == 0:
        return torch.empty((0, 5), device=bboxes.device), torch.empty((0,), device=bboxes.device)

    # Step 3: NMS
    keep = nms(filtered_bboxes, filtered_scores, iou_threshold)
    keep = keep[:max_num]

    keep_bboxes = filtered_bboxes[keep]
    keep_scores = filtered_scores[keep].unsqueeze(-1)
    keep_labels = filtered_labels[keep]

    # Apply score weight and concatenate
    keep_scores = keep_scores * score_weight
    keep_bboxes = torch.cat((keep_bboxes, keep_scores), dim=1)

    return keep_bboxes, keep_labels


def sigmoid_activation_without_bg(cls_score):
    fine_cls_score = cls_score
    score_classes = fine_cls_score.sigmoid()
    return score_classes


def get_class_concept_onehot(concept_map, attribute_list, class_list):
    num_classes = len(class_list)
    num_attributes = len(attribute_list)
    # 建立属性索引
    attr2idx = {attr: i for i, attr in enumerate(attribute_list)}
    # 初始化全 0 矩阵
    onehot = torch.zeros((num_classes, num_attributes), dtype=torch.float32)
    # 类别索引
    cls2idx = {cls: i for i, cls in enumerate(class_list)}

    # 填 one-hot
    for attr, cls_group in concept_map.items():
        attr_idx = attr2idx[attr]
        for cls in cls_group:
            cls_idx = cls2idx[cls]
            onehot[cls_idx, attr_idx] = 1.0

    return onehot


def get_dist_class_concept_onehot(concept_json, attribute_list, class_list):
    """
    Generate class-attribute matrix:
        1 = positive attribute
        0 = negative attribute
       -1 = unknown (not defined)
    """
    pos_map = concept_json["pos_map_set"]
    neg_map = concept_json["neg_map_set"]

    num_classes = len(class_list)
    num_attributes = len(attribute_list)

    # lookup
    attr2idx = {attr: i for i, attr in enumerate(attribute_list)}
    cls2idx = {cls: i for i, cls in enumerate(class_list)}

    # initialize with -1 (unknown)
    onehot = torch.full((num_classes, num_attributes), -1, dtype=torch.float32)

    # assign positive labels
    for cls, attrs in pos_map.items():
        cls_idx = cls2idx[cls]
        for attr in attrs:
            if attr in attr2idx:
                onehot[cls_idx][attr2idx[attr]] = 1.0

    # assign negative labels
    for cls, attrs in neg_map.items():
        cls_idx = cls2idx[cls]
        for attr in attrs:
            if attr in attr2idx:
                assert onehot[cls_idx][attr2idx[attr]] == -1, f'{cls} {attr} is not supported'
                onehot[cls_idx][attr2idx[attr]] = 0.0

    return onehot


def filter_bbox_iou_only(
        fix_anchors,
        fg_det_bboxes,
        iou_thr=0.5
):
    iou = box_iou(fix_anchors, fg_det_bboxes)

    remove_mask = (iou > iou_thr).any(dim=1)

    keep_mask = ~remove_mask
    keep_anchors = fix_anchors[keep_mask]

    return keep_anchors, keep_mask


class ITOWConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 0,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        self.shared_att_convs, self.shared_att_fcs, last_layer_att_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_att_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add att specific branch
        self.att_convs, self.att_fcs, self.att_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            # self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part

        x_att = x

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # shared att part
        if self.num_shared_convs > 0:
            for conv in self.shared_att_convs:
                x_att = conv(x_att)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x_att = self.avg_pool(x_att)

            x_att = x_att.flatten(1)

            for fc in self.shared_att_fcs:
                x_att = self.relu(fc(x_att))

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.att_convs:
            x_att = conv(x_att)
        if x_att.dim() > 2:
            if self.with_avg_pool:
                x_att = self.avg_pool(x_att)
            x_att = x_att.flatten(1)
        for fc in self.att_fcs:
            x_att = self.relu(fc(x_att))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return x_cls, x_att, bbox_pred
        # return cls_score, bbox_pred


@MODELS.register_module()
class ITOWShared2FCBBoxHead(ITOWConvFCBBoxHead):

    def __init__(self, owod_cfg, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.owod_cfg = owod_cfg

        known_embeddings = torch.load(owod_cfg['known_text_embeddings_path'])
        self.known_texts = known_embeddings['texts']

        # concept feat
        con_concept_feat = torch.load(owod_cfg['con_concept_feat_path'], map_location='cpu')
        con_concept_feat_list = con_concept_feat["concept_info"]['all_concept_list']
        con_concept_feat_map = con_concept_feat["concept_info"]['concept_map']

        dist_concept_feat = torch.load(owod_cfg['dist_concept_feat_path'], map_location='cpu')
        dist_concept_feat_list = dist_concept_feat['concept_info']['all_concept_list']
        dist_concept_feat_map = dist_concept_feat["concept_info"]['concept_map']

        self.con_concept_label_onehot = get_class_concept_onehot(con_concept_feat_map, con_concept_feat_list,
                                                                 self.known_texts)
        self.dist_concept_label_onehot = get_dist_class_concept_onehot(dist_concept_feat_map, dist_concept_feat_list,
                                                                       self.known_texts)

        self.focal_loss = FocalLoss()

    def loss_and_target_concept(self,
                                cls_score: Tensor,
                                concept_score: Tensor,
                                bbox_pred: Tensor,
                                rois: Tensor,
                                sampling_results: List[SamplingResult],
                                rcnn_train_cfg: ConfigDict,
                                concat: bool = True,
                                reduction_override: Optional[str] = None) -> dict:

        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss_concept(
            cls_score,
            concept_score,
            bbox_pred,
            rois,
            labels, label_weights, bbox_targets, bbox_weights,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=(labels, label_weights, bbox_targets, bbox_weights))

    def predict_by_feat_concept(self,
                                rois: Tuple[Tensor],
                                cls_scores: Tuple[Tensor],
                                concept_score: Tuple[Tensor],
                                concept_feat: Tuple[Tensor],
                                fix_concept_feat: Tuple[Tensor],
                                fix_anchors_list: List[Tensor],
                                bbox_preds: Tuple[Tensor],
                                batch_img_metas: List[dict],
                                rcnn_test_cfg: Optional[ConfigDict] = None,
                                rescale: bool = False) -> InstanceList:

        assert len(concept_score) == len(bbox_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single_concept(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                concept_score=concept_score[img_id],
                concept_feat=concept_feat[img_id],
                fix_concept_feat=fix_concept_feat[img_id],
                fix_anchors=fix_anchors_list[img_id],
                bbox_pred=bbox_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)

        return result_list

    def loss_concept(self,
                     cls_score,
                     concept_score: Tensor,
                     bbox_pred: Tensor,
                     rois: Tensor,
                     labels: Tensor,
                     label_weights: Tensor,
                     bbox_targets: Tensor,
                     bbox_weights: Tensor,
                     reduction_override: Optional[str] = None) -> dict:

        losses = dict()

        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        num_preds = cls_score.shape[0]

        # concept loss
        if self.con_concept_label_onehot.device != concept_score.device:
            self.con_concept_label_onehot = self.con_concept_label_onehot.to(concept_score.device)
            self.dist_concept_label_onehot = self.dist_concept_label_onehot.to(concept_score.device)

        _, num_con_attributes = self.con_concept_label_onehot.shape
        _, num_dist_attributes = self.dist_concept_label_onehot.shape
        N = labels.size(0)
        # 初始化全0，背景天然是0
        con_concept_onehot_label = torch.zeros(N, num_con_attributes, device=labels.device, dtype=torch.float32)
        dist_concept_onehot_label = torch.zeros(N, num_dist_attributes, device=labels.device, dtype=torch.float32)
        # 前景部分填入 one-hot
        con_concept_onehot_label[pos_inds] = self.con_concept_label_onehot[labels[pos_inds]].detach()
        dist_concept_onehot_label[pos_inds] = self.dist_concept_label_onehot[labels[pos_inds]].detach()

        con_concept_score = concept_score[:, :num_con_attributes]
        dist_concept_score = concept_score[:, num_con_attributes:]

        loss_concept_con = self.focal_loss(con_concept_score, con_concept_onehot_label, reduction_override="mean")

        dist_loss_mask = dist_concept_onehot_label != -1
        dist_concept_onehot_label[dist_loss_mask == -1] = 0

        loss_concept_dist = self.focal_loss(
            dist_concept_score,
            dist_concept_onehot_label,
            reduction_override="none"
        )

        loss_concept_dist = loss_concept_dist * dist_loss_mask  # masked tensor
        loss_concept_dist_m = loss_concept_dist.sum() / dist_loss_mask.sum()

        losses['loss_concept_con'] = loss_concept_con * 80.0
        losses['loss_concept_dist'] = loss_concept_dist_m * 80.0

        # cls loss
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:

            known_cls_score_wbg = cls_score[:, :self.num_classes + 1]
            obj_cls_score = cls_score[:, self.num_classes + 1:self.num_classes + 2]

            loss_cls_ = self.loss_cls(
                known_cls_score_wbg,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            if isinstance(loss_cls_, dict):
                losses.update(loss_cls_)
            else:
                losses['loss_cls'] = loss_cls_
            if self.custom_activation:
                acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                losses.update(acc_)
            else:
                losses['acc'] = accuracy(cls_score, labels)

            # --------------------------------------------------------
            # 3️⃣ 将已知标签映射到超类 (真实匹配)
            # --------------------------------------------------------
            obj_assigned_labels = torch.zeros((num_preds, 1),
                                              device=labels.device,
                                              dtype=torch.float32)

            # ✅ 用布尔索引替代 for 循环
            valid_mask = (labels > -1) & (labels < bg_class_ind)
            obj_assigned_labels[valid_mask, 0] = 1.0

            obj_label_weights = label_weights.unsqueeze(1).expand(-1, obj_cls_score.size(1))
            loss_cls_obj_ = self.loss_cls(
                obj_cls_score,
                obj_assigned_labels,
                obj_label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

            losses['loss_cls_obj'] = loss_cls_obj_

        return losses

    def _predict_by_feat_single_concept(
            self,
            roi: Tensor,
            cls_score: Tensor,
            concept_score: Tensor,
            concept_feat: Tensor,
            fix_concept_feat: Tensor,
            fix_anchors: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:

        # concept score
        _, num_con_attributes = self.con_concept_label_onehot.shape
        _, num_dist_attributes = self.dist_concept_label_onehot.shape

        con_concept_score = concept_score[:, :num_con_attributes]
        con_concept_score = sigmoid_activation_without_bg(con_concept_score)

        # obj_cls_score = cls_score[:, self.num_classes + 1:self.num_classes + 2]
        cls_score = cls_score[:, :self.num_classes + 1]

        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        det_bboxes, det_labels = multiclass_nms(
            bboxes,
            scores,
            rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms,
            rcnn_test_cfg.max_per_img,
            box_dim=box_dim)

        # concept bg feat
        fg_det_bboxes = det_bboxes[det_bboxes[:, -1] > 0.5][:, :4]
        fix_anchors_filter, fix_keep_mask = filter_bbox_iou_only(fix_anchors, fg_det_bboxes)
        fix_anchors_feat = fix_concept_feat[fix_keep_mask]
        bboxes_filter, bboxes_mask = filter_bbox_iou_only(bboxes, fg_det_bboxes)
        bboxes_concept_feat = concept_feat[bboxes_mask]
        fix_anchors_feat = torch.cat([fix_anchors_feat, bboxes_concept_feat], dim=0)

        concept_fg_score = get_fg_score_pca_no_center(
            fix_anchors_feat,
            bboxes_concept_feat,
            n_components=8
        ).unsqueeze(-1)
        det_bg_concept_bboxes, det_bg_concept_labels = class_agnostic_nms_with_size(bboxes_filter, concept_fg_score,
                                                                                    min_size=10,
                                                                                    iou_threshold=0.5,
                                                                                    score_threshold=0.1,
                                                                                    score_weight=0.8, max_num=200)

        det_con_concept_bboxes, det_con_concept_labels = class_agnostic_nms(bboxes, con_concept_score, max_num=80,
                                                                            iou_threshold=0.4)

        det_con_concept_labels = det_con_concept_labels + self.num_classes
        det_bg_concept_labels = det_bg_concept_labels + self.num_classes

        det_bg_concept_bboxes, det_bg_concept_labels = filter_by_ref_keep_bboxes(
            det_bg_concept_bboxes,
            det_bg_concept_labels,
            det_con_concept_bboxes,
            iou_threshold=0.5,
        )

        unknown_bboxes = torch.cat(
            [det_con_concept_bboxes, det_bg_concept_bboxes],
            dim=0)
        unknown_labels = torch.cat(
            [det_con_concept_labels, det_bg_concept_labels],
            dim=0)

        unknown_bboxes = unknown_bboxes[:100]
        unknown_labels = unknown_labels[:100]

        combined_bboxes = torch.cat(
            [det_bboxes, unknown_bboxes],
            dim=0)
        combined_labels = torch.cat(
            [det_labels, unknown_labels],
            dim=0)

        results.bboxes = combined_bboxes[:, :-1]
        results.scores = combined_bboxes[:, -1]
        results.labels = combined_labels

        return results
