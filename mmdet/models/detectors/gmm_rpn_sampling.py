"""GMM and background proposal sampling utilities used by FasterIPOW."""

import numpy as np
import torch
from mmengine.structures import InstanceData
from torchvision.ops import box_iou
from torchvision.ops import nms


def filter_gmm_boxes_with_rpn(rpn_bboxes, gmm_boxes, gmm_labels,
                              gmm_scores, iou_thr=0.5,
                              chunk_size=2048):
    """Keep valid GMM boxes that do not overlap RPN proposals too much."""
    if rpn_bboxes.numel() == 0 or gmm_boxes.numel() == 0:
        return gmm_boxes, gmm_labels, gmm_scores

    min_xy = rpn_bboxes[:, :2].min(dim=0).values
    max_xy = rpn_bboxes[:, 2:].max(dim=0).values
    in_range_mask = (
        (gmm_boxes[:, 0] >= min_xy[0]) &
        (gmm_boxes[:, 1] >= min_xy[1]) &
        (gmm_boxes[:, 2] <= max_xy[0]) &
        (gmm_boxes[:, 3] <= max_xy[1]))
    gmm_boxes = gmm_boxes[in_range_mask]
    gmm_labels = gmm_labels[in_range_mask]
    gmm_scores = gmm_scores[in_range_mask]
    if gmm_boxes.numel() == 0:
        return gmm_boxes, gmm_labels, gmm_scores

    # Avoid materializing an [num_rpn, num_gmm, 4] tensor.  Chunking also
    # keeps memory bounded when the number of classes or samples increases.
    max_ious = []
    for start in range(0, gmm_boxes.size(0), chunk_size):
        chunk = gmm_boxes[start:start + chunk_size]
        max_ious.append(box_iou(chunk, rpn_bboxes).amax(dim=1))
    keep_mask = torch.cat(max_ious, dim=0) <= iou_thr
    return (gmm_boxes[keep_mask], gmm_labels[keep_mask],
            gmm_scores[keep_mask])


def _sample_gmm(gmm, num_samples, rng):
    """Sample from a fitted sklearn GMM without reusing its serialized seed."""
    components = rng.choice(len(gmm.weights_), size=num_samples,
                            p=gmm.weights_)
    samples = np.empty((num_samples, gmm.means_.shape[1]), dtype=np.float64)
    for component in np.unique(components):
        indices = np.flatnonzero(components == component)
        samples[indices] = rng.multivariate_normal(
            gmm.means_[component], gmm.covariances_[component],
            size=len(indices))
    return samples


def sample_all_categories_from_model(models, num_per_class=50,
                                     img_shape=(800, 1333), rng=None):
    """Sample normalized GMM boxes and decode them to image pixel boxes."""
    img_h, img_w = img_shape
    rng = np.random.default_rng() if rng is None else rng
    all_proposals = []
    for gmm in models.values():
        samples = _sample_gmm(gmm, num_per_class, rng)
        if samples.size == 0:
            continue
        # Decode all samples in one NumPy operation instead of looping over
        # every box in Python.  The GMM coordinate convention is unchanged.
        widths = np.exp(samples[:, 2])
        heights = np.exp(samples[:, 3])
        boxes = np.column_stack((
            (samples[:, 0] - widths / 2) * img_w,
            (samples[:, 1] - heights / 2) * img_h,
            (samples[:, 0] + widths / 2) * img_w,
            (samples[:, 1] + heights / 2) * img_h,
        ))
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
        valid = ((boxes[:, 2] > boxes[:, 0]) &
                 (boxes[:, 3] > boxes[:, 1]))
        if np.any(valid):
            all_proposals.append(boxes[valid])
    if not all_proposals:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(all_proposals, axis=0).astype(np.float32, copy=False)


def build_gmm_rpn_results(rpn_results_list, data_samples, gmm_models,
                          num_per_class, iou_thresh):
    """Augment each image's RPN proposals using that image's shape."""
    gmm_rpn_results = []
    for rpn_result, data_sample in zip(rpn_results_list, data_samples):
        img_shape = tuple(data_sample.metainfo['img_shape'][:2])
        sampled_boxes = sample_all_categories_from_model(
            gmm_models, num_per_class=num_per_class, img_shape=img_shape)
        device = rpn_result.bboxes.device
        gmm_boxes = torch.as_tensor(sampled_boxes,
                                    dtype=rpn_result.bboxes.dtype,
                                    device=device)
        gmm_labels = torch.zeros(gmm_boxes.size(0),
                                 dtype=rpn_result.labels.dtype, device=device)
        gmm_scores = torch.ones(gmm_boxes.size(0),
                                dtype=rpn_result.scores.dtype, device=device)
        if gmm_boxes.size(0) > 0:
            keep = nms(gmm_boxes, gmm_scores, iou_thresh)
            gmm_boxes, gmm_labels, gmm_scores = (
                gmm_boxes[keep], gmm_labels[keep], gmm_scores[keep])
        gmm_boxes, gmm_labels, gmm_scores = filter_gmm_boxes_with_rpn(
            rpn_result.bboxes, gmm_boxes, gmm_labels, gmm_scores, iou_thr=0.8)
        result = InstanceData()
        result.bboxes = torch.cat([rpn_result.bboxes, gmm_boxes], dim=0)
        result.labels = torch.cat([rpn_result.labels, gmm_labels], dim=0)
        result.scores = torch.cat([rpn_result.scores, gmm_scores], dim=0)
        gmm_rpn_results.append(result)
    return gmm_rpn_results


def generate_background_tiling_anchors(img_shape, scales=(40, 80, 120, 160, 200),
                                       overlap_ratio=0.1):
    """Generate background reference boxes for one resized image."""
    img_h, img_w = img_shape
    all_anchors = []
    for scale in scales:
        stride = max(1, int(scale * (1.0 - overlap_ratio)))
        x_coords = np.arange(0, img_w, stride)
        y_coords = np.arange(0, img_h, stride)
        xmins, ymins = np.meshgrid(x_coords, y_coords)
        anchors = np.column_stack((xmins.ravel(), ymins.ravel(),
                                   xmins.ravel() + scale,
                                   ymins.ravel() + scale))
        anchors[:, [0, 2]] = np.clip(anchors[:, [0, 2]], 0, img_w)
        anchors[:, [1, 3]] = np.clip(anchors[:, [1, 3]], 0, img_h)
        keep = ((anchors[:, 2] - anchors[:, 0] > scale / 2) &
                (anchors[:, 3] - anchors[:, 1] > scale / 2))
        all_anchors.append(anchors[keep])
    return torch.from_numpy(np.vstack(all_anchors))


def filter_fix_boxes_with_rpn(rpn_bboxes, fix_anchors):
    """Keep background reference boxes that lie within RPN proposal bounds."""
    if rpn_bboxes.numel() == 0 or fix_anchors.numel() == 0:
        return fix_anchors.detach().clone()
    min_xy = rpn_bboxes[:, :2].min(dim=0).values
    max_xy = rpn_bboxes[:, 2:].max(dim=0).values
    in_range_mask = (
        (fix_anchors[:, 0] >= min_xy[0]) &
        (fix_anchors[:, 1] >= min_xy[1]) &
        (fix_anchors[:, 2] <= max_xy[0]) &
        (fix_anchors[:, 3] <= max_xy[1]))
    return fix_anchors[in_range_mask].detach().clone()
