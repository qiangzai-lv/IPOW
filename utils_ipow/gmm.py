"""Fit per-class GMM proposal priors for IPOW.

The detector samples the four-dimensional representation
``(cx, cy, log(w), log(h))`` where ``cx, cy, w, h`` are normalized by the
image width and height.  This module writes exactly that representation and
does not create semantic-tree or superclass models.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from sklearn.mixture import GaussianMixture


TASK_CLASS_NAMES = {
    'MOWODB': {
        1: ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        2: ('truck', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator'),
        3: ('frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'),
        4: ('bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl'),
    },
    'SOWODB': {
        1: ('aeroplane', 'bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
            'cow', 'dog', 'horse', 'motorbike', 'sheep', 'train', 'elephant',
            'bear', 'zebra', 'giraffe', 'truck', 'person'),
        2: ('traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'chair', 'diningtable', 'pottedplant', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'bed', 'toilet', 'sofa'),
        3: ('frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'),
        4: ('laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'tvmonitor', 'bottle'),
    },
    'nuOWODB': {
        1: ('vehicle.bicycle', 'vehicle.motorcycle', 'vehicle.car',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck',
            'vehicle.emergency.ambulance', 'vehicle.emergency.police',
            'vehicle.construction', 'vehicle.trailer'),
        2: ('human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.wheelchair', 'human.pedestrian.stroller',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.construction_worker'),
        3: ('movable_object.barrier', 'movable_object.trafficcone',
            'movable_object.pushable_pullable', 'movable_object.debris',
            'static_object.bicycle_rack', 'animal'),
    },
}

_ALIASES = {
    'airplane': 'aeroplane',
    'dining table': 'diningtable',
    'motorcycle': 'motorbike',
    'potted plant': 'pottedplant',
    'couch': 'sofa',
    'tv': 'tvmonitor',
}


def task_class_names(dataset: str, task: int) -> tuple[str, ...]:
    dataset = dataset.strip()
    if dataset not in TASK_CLASS_NAMES or task not in TASK_CLASS_NAMES[dataset]:
        valid = ', '.join(f'{name}: {sorted(tasks)}'
                          for name, tasks in TASK_CLASS_NAMES.items())
        raise ValueError(f'Unsupported dataset/task {dataset!r}/{task}. Valid: {valid}')
    return TASK_CLASS_NAMES[dataset][task]


def bbox_to_gmm_features(boxes: np.ndarray,
                         image_width: float,
                         image_height: float,
                         eps: float = 1e-6) -> np.ndarray:
    """Convert ``xyxy`` pixel boxes to IPOW's normalized GMM coordinates."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if image_width <= 0 or image_height <= 0:
        raise ValueError('Image dimensions must be positive')
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], eps)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], eps)
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / image_width
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / image_height
    return np.column_stack((cx, cy,
                            np.log(widths / image_width),
                            np.log(heights / image_height)))


def _xml_text(node: ET.Element, path: Path) -> str:
    if node is None or node.text is None:
        raise ValueError(f'Missing XML field in {path}')
    return node.text.strip()


def load_task_bbox_features(data_root: str | Path,
                            dataset: str,
                            task: int,
                            image_set: str = 'train',
                            annotation_dir: str = 'Annotations') -> Dict[str, np.ndarray]:
    """Load normalized bbox features for only the classes introduced in task."""
    classes = set(task_class_names(dataset, task))
    root = Path(data_root)
    split = root / 'ImageSets' / dataset / f't{task}_{image_set}.txt'
    ann_root = root / annotation_dir / dataset
    if not split.is_file():
        raise FileNotFoundError(f'Split file not found: {split}')

    by_class: Dict[str, list[np.ndarray]] = {name: [] for name in classes}
    image_ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    for image_id in image_ids:
        xml_path = ann_root / f'{image_id}.xml'
        if not xml_path.is_file():
            warnings.warn(f'Skipping missing annotation: {xml_path}')
            continue
        tree = ET.parse(xml_path)
        size = tree.getroot().find('size')
        width = float(_xml_text(size.find('width'), xml_path))
        height = float(_xml_text(size.find('height'), xml_path))
        for obj in tree.getroot().findall('object'):
            raw_name = _xml_text(obj.find('name'), xml_path)
            name = _ALIASES.get(raw_name, raw_name)
            if name not in classes:
                continue
            box = obj.find('bndbox')
            coords = [float(_xml_text(box.find(key), xml_path)) - 1.0 for key in
                      ('xmin', 'ymin', 'xmax', 'ymax')]
            by_class[name].append(bbox_to_gmm_features([coords], width, height)[0])
    return {name: np.asarray(values, dtype=np.float64).reshape(-1, 4)
            for name, values in by_class.items() if values}


def fit_task_gmms(class_features: Mapping[str, np.ndarray],
                  n_components: int = 8,
                  random_state: int = 42,
                  reg_covar: float = 1e-5,
                  output_path: str | Path | None = None) -> dict:
    """Fit one GMM per supplied category and optionally save the IPOW payload."""
    if n_components < 1:
        raise ValueError('n_components must be >= 1')
    models = {}
    sample_counts = {}
    for class_name, features in class_features.items():
        features = np.asarray(features, dtype=np.float64).reshape(-1, 4)
        if len(features) == 0:
            continue
        components = min(n_components, len(features))
        model = GaussianMixture(
            n_components=components,
            covariance_type='full',
            reg_covar=reg_covar,
            random_state=random_state,
            n_init=3,
        )
        model.fit(features)
        models[class_name] = model
        sample_counts[class_name] = len(features)

    payload = {
        'models': models,
        'metadata': {
            'feature_format': 'normalized_cx_cy_log_w_log_h',
            'n_components': n_components,
            'random_state': random_state,
            'sample_counts': sample_counts,
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('wb') as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit IPOW per-class GMM proposal priors')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--dataset', choices=sorted(TASK_CLASS_NAMES), required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--image-set', default='train')
    parser.add_argument('--annotation-dir', default='Annotations')
    parser.add_argument('--n-components', type=int, default=8)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    features = load_task_bbox_features(
        args.data_root, args.dataset, args.task, args.image_set, args.annotation_dir)
    payload = fit_task_gmms(features, args.n_components, args.random_state,
                             output_path=args.output)
    print(f'Saved {len(payload["models"])} class GMMs to {args.output}')
    for name, count in payload['metadata']['sample_counts'].items():
        print(f'{name}: {count} boxes')


if __name__ == '__main__':
    main()
