# Copyright (c) OpenMMLab. All rights reserved.
import copy

import joblib
import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor
from torchvision.ops import nms

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


def filter_gmm_boxes_with_rpn(rpn_bboxes, gmm_boxes, gmm_labels, gmm_scores, iou_thr=0.5):
    """
    Args:
        rpn_bboxes: Tensor [Nr, 4]  -- RPN 提议框
        gmm_boxes: Tensor [Ng, 4]  -- GMM 目标框
        iou_thr: float             -- IoU 阈值（删掉 IoU > thr 的 GMM 框）

    Returns:
        gmm_filtered: 过滤后的 gmm_boxes
    """

    # ------------------------
    # Step 1: 删除超界框
    # ------------------------

    # RPN 外接范围
    min_xy = rpn_bboxes[:, :2].min(dim=0).values  # [min_x1, min_y1]
    max_xy = rpn_bboxes[:, 2:].max(dim=0).values  # [max_x2, max_y2]

    # 超界筛除 mask
    in_range_mask = (
            (gmm_boxes[:, 0] >= min_xy[0]) &
            (gmm_boxes[:, 1] >= min_xy[1]) &
            (gmm_boxes[:, 2] <= max_xy[0]) &
            (gmm_boxes[:, 3] <= max_xy[1])
    )

    gmm_boxes = gmm_boxes[in_range_mask]
    gmm_labels = gmm_labels[in_range_mask]
    gmm_scores = gmm_scores[in_range_mask]

    # ------------------------
    # Step 2: 删除与 RPN IoU > threshold 的框
    # ------------------------

    # IoU 计算（广播方式）
    # rpn: [Nr,1,4], gmm:[1,Ng,4]
    rpn = rpn_bboxes[:, None, :]
    gmm = gmm_boxes[None, :, :]

    # 交集
    inter_x1 = torch.max(rpn[..., 0], gmm[..., 0])
    inter_y1 = torch.max(rpn[..., 1], gmm[..., 1])
    inter_x2 = torch.min(rpn[..., 2], gmm[..., 2])
    inter_y2 = torch.min(rpn[..., 3], gmm[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    # 面积
    area_rpn = (rpn[..., 2] - rpn[..., 0]) * (rpn[..., 3] - rpn[..., 1])
    area_gmm = (gmm[..., 2] - gmm[..., 0]) * (gmm[..., 3] - gmm[..., 1])

    # IoU
    union = area_rpn + area_gmm - inter
    iou = inter / union

    # 每个 GMM 框的最大 IoU
    max_iou_gmm = iou.max(dim=0).values  # shape [Ng]

    # 小于阈值的保留
    keep_mask = max_iou_gmm <= iou_thr
    gmm_boxes_filtered = gmm_boxes[keep_mask]
    gmm_labels_filtered = gmm_labels[keep_mask]
    gmm_scores_filtered = gmm_scores[keep_mask]

    return gmm_boxes_filtered, gmm_labels_filtered, gmm_scores_filtered


def sample_all_categories_from_model(models, num_per_class=50, img_shape=(800, 1333)):
    img_h, img_w = img_shape
    all_proposals = []

    # 2. 遍历每一个类别进行采样
    for label, gmm in models.items():
        # 从该类别的 GMM 中采样
        samples, _ = gmm.sample(num_per_class)

        count = 0
        for s in samples:
            cx, cy, log_w, log_h = s
            # 逆转训练时的对数变换
            w, h = np.exp(log_w), np.exp(log_h)

            # 计算像素坐标
            xmin = (cx - w / 2) * img_w
            ymin = (cy - h / 2) * img_h
            xmax = (cx + w / 2) * img_w
            ymax = (cy + h / 2) * img_h

            # 边界裁剪与合法性检查
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)

            if xmax > xmin and ymax > ymin:
                all_proposals.append([xmin, ymin, xmax, ymax])
                count += 1

    return np.array(all_proposals)


def generate_background_tiling_anchors(img_shape=(800, 1333), scales=[40, 80, 120, 160, 200], overlap_ratio=0.1):
    """
    全图平铺采样：确保背景无死角覆盖
    :param img_shape: (H, W)
    :param scales: 尺度列表，例如 [64, 128, 256]
    :param overlap_ratio: 重叠比例。0.0 表示刚好平铺（不重叠），0.5 表示每个框重叠一半。
    """
    img_h, img_w = img_shape
    all_anchors = []

    for scale in scales:
        # 计算步长：如果不重叠，步长就等于尺度本身
        # stride = scale * (1 - overlap_ratio)
        stride = int(scale * (1.0 - overlap_ratio))
        stride = max(1, stride)

        # 生成起点坐标，确保从 0 开始覆盖到图像边缘
        # 我们让坐标从 0 开始，一直铺到图像末尾
        x_coords = np.arange(0, img_w, stride)
        y_coords = np.arange(0, img_h, stride)

        # 使用网格化生成所有左上角点
        xv, yv = np.meshgrid(x_coords, y_coords)
        xmins = xv.ravel()
        ymins = yv.ravel()

        # 计算对应的右下角
        xmaxs = xmins + scale
        ymaxs = ymins + scale

        # 封装当前尺度的所有框
        batch_anchors = np.column_stack([xmins, ymins, xmaxs, ymaxs])

        # 边界处理：由于平铺可能超出边界，我们需要决定是裁剪还是丢弃
        # 这里建议裁剪，以保证“全图覆盖”
        batch_anchors[:, [0, 2]] = np.clip(batch_anchors[:, [0, 2]], 0, img_w)
        batch_anchors[:, [1, 3]] = np.clip(batch_anchors[:, [1, 3]], 0, img_h)

        # 过滤掉太小的碎框（比如在最边缘只剩 1 像素宽的框）
        keep = (batch_anchors[:, 2] - batch_anchors[:, 0] > scale / 2) & \
               (batch_anchors[:, 3] - batch_anchors[:, 1] > scale / 2)

        all_anchors.append(batch_anchors[keep])

        print(f"尺度 {scale:3d} | 平铺步长 {stride:3d} | 产生背景块数量: {len(batch_anchors[keep])}")

    return torch.from_numpy(np.vstack(all_anchors))


def filter_fix_boxes_with_rpn(rpn_bboxes, fix_anchors):
    # RPN 外接范围
    min_xy = rpn_bboxes[:, :2].min(dim=0).values  # [min_x1, min_y1]
    max_xy = rpn_bboxes[:, 2:].max(dim=0).values  # [max_x2, max_y2]
    # 超界筛除 mask
    in_range_mask = (
            (fix_anchors[:, 0] >= min_xy[0]) &
            (fix_anchors[:, 1] >= min_xy[1]) &
            (fix_anchors[:, 2] <= max_xy[0]) &
            (fix_anchors[:, 3] <= max_xy[1])
    )
    return fix_anchors[in_range_mask].detach().clone()


@MODELS.register_module()
class FasterIPOW(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 owod_cfg: ConfigType,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.owod_cfg = owod_cfg
        # text embedding
        known_embeddings = torch.load(self.owod_cfg['known_text_embeddings_path'])
        self.known_texts = known_embeddings['texts']
        known_embeddings_tensor = known_embeddings['embeddings']  # [num_leaves, d]
        self.known_embeddings = nn.Parameter(known_embeddings_tensor, requires_grad=False)
        self.known_cls_nums = len(self.known_texts)

        con_concept_feat = torch.load(owod_cfg['con_concept_feat_path'], map_location='cpu')
        self.con_concept_embeddings = nn.Parameter(con_concept_feat["embeddings"], requires_grad=False)
        dist_concept_feat = torch.load(owod_cfg['dist_concept_feat_path'], map_location='cpu')
        self.dist_concept_embeddings = nn.Parameter(dist_concept_feat['embeddings'], requires_grad=False)

        # 随机初始化一个可训练的背景 embedding 与 前景类别
        self.background_emb = nn.Parameter(torch.randn(1, known_embeddings_tensor.size(1)), requires_grad=True)
        self.obj_emb = nn.Parameter(torch.randn(1, known_embeddings_tensor.size(1)), requires_grad=True)

        # gmm_models
        self.gmm_models = joblib.load(self.owod_cfg.randm_gmm_path)['models']
        print(f"成功加载GMM模型，共包含 {len(self.gmm_models)} 个类别。")
        self.gmm_iou_thresh = self.owod_cfg.gmm_iou_thresh
        # fix_anchors
        self.fix_anchors = generate_background_tiling_anchors()


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:

        x = self.extract_feat(batch_inputs)

        losses = dict()

        # text embedding
        all_text_embedding = torch.cat(
            [self.known_embeddings, self.background_emb, self.obj_emb], dim=0
        )

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        # GMM
        # gmm generate
        gmm_sampled_boxes = sample_all_categories_from_model(self.gmm_models, num_per_class=self.owod_cfg.gmm_n_samples)
        device = rpn_results_list[0].bboxes.device
        bbox_dtype = rpn_results_list[0].bboxes.dtype
        score_dtype = rpn_results_list[0].scores.dtype
        label_dtype = rpn_results_list[0].labels.dtype

        gmm_boxes_tensor = torch.tensor(gmm_sampled_boxes, dtype=bbox_dtype, device=device)
        gmm_labels = torch.zeros(len(gmm_boxes_tensor), dtype=label_dtype, device=device)
        gmm_scores = torch.full((len(gmm_boxes_tensor),), 1.0, dtype=score_dtype, device=device)

        keep = nms(gmm_boxes_tensor, gmm_scores, self.gmm_iou_thresh)
        gmm_boxes_tensor = gmm_boxes_tensor[keep]
        gmm_labels = gmm_labels[keep]
        gmm_scores = gmm_scores[keep]

        gmm_rpn_results = []
        for i, rpn_result in enumerate(rpn_results_list):
            # 原始 bboxes, labels, scores
            rpn_bboxes = rpn_result.bboxes
            rpn_labels = rpn_result.labels
            rpn_scores = rpn_result.scores
            gmm_rpn_result = InstanceData()
            gmm_boxes_filtered, gmm_labels_filtered, gmm_scores_filtered = filter_gmm_boxes_with_rpn(rpn_bboxes,
                                                                                                     gmm_boxes_tensor,
                                                                                                     gmm_labels,
                                                                                                     gmm_scores,
                                                                                                     iou_thr=0.8)

            gmm_rpn_result.bboxes = torch.cat([rpn_bboxes, gmm_boxes_filtered], dim=0)
            gmm_rpn_result.labels = torch.cat([rpn_labels, gmm_labels_filtered], dim=0)
            gmm_rpn_result.scores = torch.cat([rpn_scores, gmm_scores_filtered], dim=0)
            gmm_rpn_results.append(gmm_rpn_result)

        # ROI
        roi_losses = self.roi_head.loss(x, all_text_embedding, [self.con_concept_embeddings, self.dist_concept_embeddings], gmm_rpn_results, batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        x = self.extract_feat(batch_inputs)

        # text embedding
        all_text_embedding = torch.cat(
            [self.known_embeddings, self.background_emb], dim=0
        )

        # RPN
        rpn_results_list = self.rpn_head.predict(
            x, batch_data_samples, rescale=False)

        # GMM
        # gmm generate
        gmm_sampled_boxes = sample_all_categories_from_model(self.gmm_models, num_per_class=self.owod_cfg.gmm_n_samples)
        device = rpn_results_list[0].bboxes.device
        bbox_dtype = rpn_results_list[0].bboxes.dtype
        score_dtype = rpn_results_list[0].scores.dtype
        label_dtype = rpn_results_list[0].labels.dtype

        gmm_boxes_tensor = torch.tensor(gmm_sampled_boxes, dtype=bbox_dtype, device=device)
        gmm_labels = torch.zeros(len(gmm_boxes_tensor), dtype=label_dtype, device=device)
        gmm_scores = torch.full((len(gmm_boxes_tensor),), 1.0, dtype=score_dtype, device=device)

        keep = nms(gmm_boxes_tensor, gmm_scores, self.gmm_iou_thresh)
        gmm_boxes_tensor = gmm_boxes_tensor[keep]
        gmm_labels = gmm_labels[keep]
        gmm_scores = gmm_scores[keep]

        # BG FIX ANCHORS
        if self.fix_anchors.device != device:
            self.fix_anchors = self.fix_anchors.to(device=device)

        fix_anchors_list = []

        gmm_rpn_results = []
        for i, rpn_result in enumerate(rpn_results_list):
            # 原始 bboxes, labels, scores
            rpn_bboxes = rpn_result.bboxes
            rpn_labels = rpn_result.labels
            rpn_scores = rpn_result.scores
            gmm_rpn_result = InstanceData()
            gmm_boxes_filtered, gmm_labels_filtered, gmm_scores_filtered = filter_gmm_boxes_with_rpn(rpn_bboxes,
                                                                                                     gmm_boxes_tensor,
                                                                                                     gmm_labels,
                                                                                                     gmm_scores,
                                                                                                     iou_thr=0.8)
            fix_an_filter = filter_fix_boxes_with_rpn(rpn_bboxes, self.fix_anchors)
            if len(fix_an_filter) > 10:
                fix_anchors_list.append(fix_an_filter)
            else:
                fix_anchors_list.append(self.fix_anchors.clone())
            gmm_rpn_result.bboxes = torch.cat([rpn_bboxes, gmm_boxes_filtered], dim=0)
            gmm_rpn_result.labels = torch.cat([rpn_labels, gmm_labels_filtered], dim=0)
            gmm_rpn_result.scores = torch.cat([rpn_scores, gmm_scores_filtered], dim=0)
            gmm_rpn_results.append(gmm_rpn_result)

        # ROI
        results_list = self.roi_head.predict(x, all_text_embedding, [self.con_concept_embeddings, self.dist_concept_embeddings], gmm_rpn_results,
            fix_anchors_list,
            batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        for pred_sample in batch_data_samples:
            pred_sample.pred_instances.labels[
                pred_sample.pred_instances.labels >= self.known_cls_nums] = self.known_cls_nums

        return batch_data_samples
