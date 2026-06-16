# Copyright (c) OpenMMLab. All rights reserved.

import copy
import pickle

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

def filter_gmm_boxes_with_rpn(rpn_bboxes, gmm_boxes,  gmm_labels, gmm_scores, iou_thr=0.5):
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
    min_xy = rpn_bboxes[:, :2].min(dim=0).values   # [min_x1, min_y1]
    max_xy = rpn_bboxes[:, 2:].max(dim=0).values   # [max_x2, max_y2]

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

def load_gmm_models_pickle(load_path):
    """
    使用 pickle 加载 gmm_models 字典
    """
    with open(load_path, 'rb') as f:
        gmm_models = pickle.load(f)
    print(f"✅ 已加载 GMM 模型：{list(gmm_models.keys())}")
    return gmm_models


def xywh_to_xyxy(boxes):
    """[x_center, y_center, w, h] -> [x_min, y_min, x_max, y_max]"""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def sample_gmm_xywh(gmm_models, class_name, n_samples=10, clip_box=None):
    """
    从指定类别 GMM 中采样 bbox，并转换回 xyxy
    """

    gmm = gmm_models[class_name]
    samples_xywh, _ = gmm.sample(n_samples)
    sampled_boxes = xywh_to_xyxy(samples_xywh)  # 转回 xyxy
    # 确保 x_min < x_max, y_min < y_max
    x_min = np.minimum(sampled_boxes[:, 0], sampled_boxes[:, 2])
    y_min = np.minimum(sampled_boxes[:, 1], sampled_boxes[:, 3])
    x_max = np.maximum(sampled_boxes[:, 0], sampled_boxes[:, 2])
    y_max = np.maximum(sampled_boxes[:, 1], sampled_boxes[:, 3])
    sampled_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

    # 可选裁剪到图像范围
    if clip_box is not None:
        x0, y0, x1, y1 = clip_box
        sampled_boxes[:, 0] = np.clip(sampled_boxes[:, 0], x0, x1)
        sampled_boxes[:, 1] = np.clip(sampled_boxes[:, 1], y0, y1)
        sampled_boxes[:, 2] = np.clip(sampled_boxes[:, 2], x0, x1)
        sampled_boxes[:, 3] = np.clip(sampled_boxes[:, 3], y0, y1)

    return sampled_boxes


def generate_fixed_anchors(W=1333, H=800,
                           num_anchors=1000,
                           scales=(32, 64, 128, 256),
                           ratios=(0.5, 1.0, 2.0)):
    """
    在整张图片上固定采样约 num_anchors 个锚框。
    不依赖特征图，不使用随机。
    """

    # 计算每个方向上网格数量
    grid_size = int((num_anchors / len(scales) / len(ratios)) ** 0.5)
    grid_x = torch.linspace(0, W, grid_size + 2)[1:-1]  # 避免边界
    grid_y = torch.linspace(0, H, grid_size + 2)[1:-1]

    # 所有中心点组合
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
    centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [M, 2]

    anchors = []
    for s in scales:
        for r in ratios:
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            # 扩展为所有中心
            cx, cy = centers[:, 0], centers[:, 1]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            anchors.append(torch.stack([x1, y1, x2, y2], dim=-1))

    anchors = torch.cat(anchors, dim=0)

    # 限制在图像范围内
    anchors[:, 0::2] = anchors[:, 0::2].clamp(0, W - 1)
    anchors[:, 1::2] = anchors[:, 1::2].clamp(0, H - 1)

    return anchors  # torch.Size([~1000, 4])


@MODELS.register_module()
class FasterITOWGMM(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 owod_cfg: ConfigType,
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
        self.gmm_models = load_gmm_models_pickle(self.owod_cfg.randm_gmm_path)
        self.gmm_iou_thresh = self.owod_cfg.gmm_iou_thresh
        self.fix_anchors = generate_fixed_anchors(num_anchors=self.owod_cfg.num_anchors)


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:

        x = self.extract_feat(batch_inputs)

        # text embedding
        all_text_embedding = torch.cat(
            [self.known_embeddings, self.background_emb, self.obj_emb], dim=0
        )

        losses = dict()

        # RPN
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        # GMM
        # gmm generate
        gmm_gen_texts = self.gmm_models.keys()
        gmm_sampled_boxes = []
        for gmm_txt in gmm_gen_texts:
            sampled_boxes = sample_gmm_xywh(self.gmm_models, gmm_txt, n_samples=self.owod_cfg.gmm_n_samples,
                                            clip_box=(0, 0, 1333, 800))
            gmm_sampled_boxes.extend(sampled_boxes)

        device = rpn_results_list[0].bboxes.device
        bbox_dtype = rpn_results_list[0].bboxes.dtype
        score_dtype = rpn_results_list[0].scores.dtype
        label_dtype = rpn_results_list[0].labels.dtype

        gmm_boxes_tensor = torch.tensor(gmm_sampled_boxes, dtype=bbox_dtype, device=device)
        if self.fix_anchors.device != device:
            self.fix_anchors = self.fix_anchors.to(device=device, dtype=score_dtype)

        # 加入 fix anchors
        gmm_boxes_tensor = torch.cat([gmm_boxes_tensor, self.fix_anchors], dim=0)
        gmm_labels = torch.zeros(len(gmm_boxes_tensor), dtype=label_dtype, device=device)
        gmm_scores = torch.full((len(gmm_boxes_tensor),), 0.5, dtype=score_dtype, device=device)

        keep = nms(gmm_boxes_tensor, gmm_scores, self.gmm_iou_thresh)
        gmm_boxes_tensor = gmm_boxes_tensor[keep]
        gmm_labels = gmm_labels[keep]
        gmm_scores = gmm_scores[keep]

        gmm_rpn_results = []
        for rpn_result in rpn_results_list:
            # 原始 bboxes, labels, scores
            rpn_bboxes = rpn_result.bboxes
            rpn_labels = rpn_result.labels
            rpn_scores = rpn_result.scores
            gmm_rpn_result = InstanceData()
            gmm_boxes_filtered, gmm_labels_filtered, gmm_scores_filtered = filter_gmm_boxes_with_rpn(rpn_bboxes, gmm_boxes_tensor,  gmm_labels, gmm_scores, iou_thr=0.8)
            gmm_rpn_result.bboxes = torch.cat([rpn_bboxes, gmm_boxes_filtered], dim=0)
            gmm_rpn_result.labels = torch.cat([rpn_labels, gmm_labels_filtered], dim=0)
            gmm_rpn_result.scores = torch.cat([rpn_scores, gmm_scores_filtered], dim=0)
            gmm_rpn_results.append(gmm_rpn_result)

        # ROI
        roi_losses = self.roi_head.loss(x, all_text_embedding,
                                        [self.con_concept_embeddings, self.dist_concept_embeddings], gmm_rpn_results,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        x = self.extract_feat(batch_inputs)

        # text embedding
        all_text_embedding = torch.cat(
            [self.known_embeddings, self.background_emb, self.obj_emb], dim=0
        )

        # RPN
        rpn_results_list = self.rpn_head.predict(
            x, batch_data_samples, rescale=False)

        # GMM
        # gmm generate
        gmm_gen_texts = self.gmm_models.keys()
        gmm_sampled_boxes = []
        for gmm_txt in gmm_gen_texts:
            sampled_boxes = sample_gmm_xywh(self.gmm_models, gmm_txt, n_samples=self.owod_cfg.gmm_n_samples,
                                            clip_box=(0, 0, 1333, 800))
            gmm_sampled_boxes.extend(sampled_boxes)

        device = rpn_results_list[0].bboxes.device
        bbox_dtype = rpn_results_list[0].bboxes.dtype
        score_dtype = rpn_results_list[0].scores.dtype
        label_dtype = rpn_results_list[0].labels.dtype

        gmm_boxes_tensor = torch.tensor(gmm_sampled_boxes, dtype=bbox_dtype, device=device)
        if self.fix_anchors.device != device:
            self.fix_anchors = self.fix_anchors.to(device=device, dtype=score_dtype)

        # 加入 fix anchors
        gmm_boxes_tensor = torch.cat([gmm_boxes_tensor, self.fix_anchors], dim=0)
        gmm_labels = torch.zeros(len(gmm_boxes_tensor), dtype=label_dtype, device=device)
        gmm_scores = torch.full((len(gmm_boxes_tensor),), 0.5, dtype=score_dtype, device=device)

        keep = nms(gmm_boxes_tensor, gmm_scores, self.gmm_iou_thresh)
        gmm_boxes_tensor = gmm_boxes_tensor[keep]
        gmm_labels = gmm_labels[keep]
        gmm_scores = gmm_scores[keep]

        gmm_rpn_results = []
        for rpn_result in rpn_results_list:
            # 原始 bboxes, labels, scores
            rpn_bboxes = rpn_result.bboxes
            rpn_labels = rpn_result.labels
            rpn_scores = rpn_result.scores
            gmm_rpn_result = InstanceData()
            gmm_boxes_filtered, gmm_labels_filtered, gmm_scores_filtered = filter_gmm_boxes_with_rpn(rpn_bboxes, gmm_boxes_tensor,  gmm_labels, gmm_scores, iou_thr=0.8)
            gmm_rpn_result.bboxes = torch.cat([rpn_bboxes, gmm_boxes_filtered], dim=0)
            gmm_rpn_result.labels = torch.cat([rpn_labels, gmm_labels_filtered], dim=0)
            gmm_rpn_result.scores = torch.cat([rpn_scores, gmm_scores_filtered], dim=0)
            gmm_rpn_results.append(gmm_rpn_result)

        # ROI
        results_list = self.roi_head.predict(
            x, all_text_embedding,
            [self.con_concept_embeddings, self.dist_concept_embeddings],
            gmm_rpn_results,
            batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        for pred_sample in batch_data_samples:
            pred_sample.pred_instances.labels[
                pred_sample.pred_instances.labels >= self.known_cls_nums] = self.known_cls_nums

        return batch_data_samples

    def set_epoch(self, epoch):
        self.roi_head.bbox_head.epoch = epoch
