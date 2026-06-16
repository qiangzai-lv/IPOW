import argparse
import os
import time

import torch
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector

coco_cls = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class MmdetModel:
    def __init__(self, cfg_path, pt_path, skip_scores=0.5):
        self.cfg_path = cfg_path
        self.pt_path = pt_path
        self.skip_scores = skip_scores
        self.class_names = coco_cls

        self.model = init_detector(self.cfg_path, self.pt_path)

    def predict(self, img_path):
        result = inference_detector(self.model, img_path)

        labels = result.pred_instances.labels
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores

        # 简单过滤（保持与你原逻辑一致）
        keep = scores > self.skip_scores
        labels = labels[keep]
        bboxes = bboxes[keep]

        known_index = labels < len(self.class_names)
        labels = labels[known_index]
        bboxes = bboxes[known_index]

        return bboxes  # 不需要返回具体结果


def load_images(img_dir):
    img_paths = []

    for file_name in os.listdir(img_dir):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_paths.append(os.path.join(img_dir, file_name))

    img_paths.sort()  # 保持顺序一致（可复现）

    return img_paths


def main(args):
    model = MmdetModel(args.mmdet_cfg, args.mmdet_pt)

    img_paths = load_images(args.input_images)
    print(f"Total images: {len(img_paths)}")

    # 🔥 预热（非常重要，避免第一次慢）
    print("Warming up...")
    for i in range(min(10, len(img_paths))):
        _ = model.predict(img_paths[i])

    torch.cuda.synchronize()

    # 🔥 开始计时
    start_time = time.time()

    for img_path in tqdm(img_paths):
        _ = model.predict(img_path)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = len(img_paths) / total_time

    print("\n===== Speed Test Result =====")
    print(f"Total images: {len(img_paths)}")
    print(f"Total time: {total_time:.4f} s")
    print(f"FPS: {fps:.2f}")
    print(f"Latency per image: {1000 / fps:.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser("MMDet Inference Speed Test")

    parser.add_argument("--input_images", type=str, default="data/TestImages")
    parser.add_argument("--mmdet_cfg", type=str, default="configs/itow/itow_owod_mowodb_t3.py")
    parser.add_argument("--mmdet_pt", type=str, default="work_dirs/itow_owod_mowodb_t1/epoch_7.pth")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)