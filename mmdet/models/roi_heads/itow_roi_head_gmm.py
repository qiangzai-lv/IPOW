# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from .base_roi_head import BaseRoIHead
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances


def orthogonal_loss(con_concept_feat, dist_concept_feat):
    dot = (con_concept_feat @ dist_concept_feat.T).diagonal()
    con_norm = con_concept_feat.norm(dim=1)
    dist_norm = dist_concept_feat.norm(dim=1)
    cos_sim = dot / (con_norm * dist_norm + 1e-6)
    loss = (cos_sim ** 2).mean()
    return loss


class ContrastiveHead(BaseModule):

    def __init__(self,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum
        # ✅ 新增线性投影层 (图像特征)
        self.proj = nn.Linear(1024, 512)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.proj(x)  # [N, 512]
        x = F.normalize(x, dim=-1, p=2)  # 单位化
        w = F.normalize(w, dim=-1, p=2)  # [K, 512]
        logits = torch.matmul(x, w.T)  # [N, K]
        logits = logits * self.logit_scale.exp() + self.bias
        return logits, x


class ConceptContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    """

    def __init__(self,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum
        # ✅ 新增线性投影层 (图像特征)
        self.proj = nn.Linear(1024, 512)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """Forward function of contrastive learning.

        Args:
            x: Tensor of shape [N, C]
            w: Tensor of shape [B, K, C] or [K, C]
        Returns:
            similarity map or matrix
        """
        x = self.proj(x)  # [N, 512]
        x = F.normalize(x, dim=-1, p=2)  # 单位化
        w = F.normalize(w, dim=-1, p=2)  # [K, 512]
        logits = torch.matmul(x, w.T)  # [N, K]
        logits = logits * self.logit_scale.exp() + self.bias
        return logits, x


@MODELS.register_module()
class ITOWGMMRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cls_contrasts_head = ContrastiveHead()
        self.con_concept_contrasts_head = ConceptContrastiveHead()
        self.dist_concept_contrasts_head = ConceptContrastiveHead()

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = MODELS.build(mask_head)

    def loss(self, x: Tuple[Tensor], text_feat, concept_embedding, rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:

        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        bbox_results = self.bbox_loss(x, text_feat, concept_embedding, sampling_results)
        losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], text_feat, concept_embedding, rois: Tensor):

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, concept_feat, bbox_pred = self.bbox_head(bbox_feats)

        cls_score, cls_score_feat = self.cls_contrasts_head(cls_score, text_feat)

        con_concept_embedding = concept_embedding[0]
        dist_concept_embedding = concept_embedding[1]

        con_concept_score, con_concept_feat = self.con_concept_contrasts_head(concept_feat, con_concept_embedding)
        dist_concept_score, dist_concept_feat = self.dist_concept_contrasts_head(concept_feat, dist_concept_embedding)

        concept_feat = F.normalize(concept_feat, dim=-1, p=2)  # [N, 36, 512]

        concept_score = torch.cat([con_concept_score, dist_concept_score], dim=-1)

        # concept_orth_loss
        concept_orth_loss = orthogonal_loss(con_concept_feat, dist_concept_feat)

        bbox_results = dict(
            concept_score=concept_score,
            concept_feat=concept_feat,
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, concept_orth_loss

    def bbox_loss(self, x: Tuple[Tensor],
                  text_feat: Tensor,
                  concept_embedding: Tensor,
                  sampling_results: List[SamplingResult]) -> dict:

        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results, concept_orth_loss = self._bbox_forward(x, text_feat, concept_embedding, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target_concept(
            concept_score=bbox_results['concept_score'],
            concept_feat=bbox_results['concept_feat'],
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_loss_and_target['loss_bbox']['concept_orth_loss'] = concept_orth_loss

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     text_feat,
                     concept_embedding,
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:

        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results, _ = self._bbox_forward(x, text_feat, concept_embedding, rois)

        # split batch bbox prediction back to each image
        concept_feat = bbox_results['concept_feat']
        concept_score = bbox_results['concept_score']
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)
        concept_feat = concept_feat.split(num_proposals_per_img, 0)
        concept_score = concept_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None,) * len(proposals)

        result_list = self.bbox_head.predict_by_feat_concept(
            rois=rois,
            concept_score=concept_score,
            concept_feat=concept_feat,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def predict(self,
                x: Tuple[Tensor],
                text_feat,
                concept_embedding,
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            text_feat,
            concept_embedding,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        return results_list
