# Copyright (c) OpenMMLab. All rights reserved.
import copy

import joblib
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .gmm_rpn_sampling import (build_gmm_rpn_results,
                               filter_fix_boxes_with_rpn,
                               generate_background_tiling_anchors)
from .two_stage import TwoStageDetector


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
        # concept embedding
        con_concept_feat = torch.load(owod_cfg['con_concept_feat_path'], map_location='cpu')
        self.con_concept_embeddings = nn.Parameter(con_concept_feat["embeddings"], requires_grad=False)
        dist_concept_feat = torch.load(owod_cfg['dist_concept_feat_path'], map_location='cpu')
        self.dist_concept_embeddings = nn.Parameter(dist_concept_feat['embeddings'], requires_grad=False)

        # 随机初始化一个可训练的背景 embedding 与 前景类别
        self.background_emb = nn.Parameter(torch.randn(1, known_embeddings_tensor.size(1)), requires_grad=True)
        # gmm_models
        self.gmm_models = joblib.load(self.owod_cfg.randm_gmm_path)['models']
        print(f"成功加载GMM模型，共包含 {len(self.gmm_models)} 个类别。")
        self.gmm_iou_thresh = self.owod_cfg.gmm_iou_thresh


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

        gmm_rpn_results = build_gmm_rpn_results(
            rpn_results_list, batch_data_samples, self.gmm_models,
            self.owod_cfg.gmm_n_samples, self.gmm_iou_thresh)

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

        device = rpn_results_list[0].bboxes.device
        fix_anchors_list = []

        gmm_rpn_results = build_gmm_rpn_results(
            rpn_results_list, batch_data_samples, self.gmm_models,
            self.owod_cfg.gmm_n_samples, self.gmm_iou_thresh)
        for data_sample, rpn_result in zip(batch_data_samples, rpn_results_list):
            rpn_bboxes = rpn_result.bboxes
            img_shape = tuple(data_sample.metainfo['img_shape'][:2])
            fix_anchors = generate_background_tiling_anchors(img_shape).to(
                device=device, dtype=rpn_bboxes.dtype)
            fix_an_filter = filter_fix_boxes_with_rpn(rpn_bboxes, fix_anchors)
            if len(fix_an_filter) > 10:
                fix_anchors_list.append(fix_an_filter)
            else:
                fix_anchors_list.append(fix_anchors)

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
