import json
from collections import defaultdict

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from mmdet.registry import DATASETS
import pickle

def generate_sublist_full(tree, unknown_info, level=0):
    """
    对三层树递归生成 unknown 向量，并记录每个节点的层级
    tree: dict 或 list
    unknown_info: 用于存储每个节点的叶子信息及层级的字典
    level: 当前的层级
    """
    if isinstance(tree, list):
        # 到叶子层，返回叶子列表
        return tree
    elif isinstance(tree, dict):
        all_leaves = []
        for node_name, subtree in tree.items():
            # 递归获取该子节点下的所有叶子
            leaves = generate_sublist_full(subtree, unknown_info, level + 1)
            all_leaves.extend(leaves)

            # 为当前子类（中间层）生成 unknown 向量，并记录层级信息
            if leaves:
                unknown_info[f"{node_name}"] = {
                    'leaves': leaves,
                    'level': level
                }
        return all_leaves
    return None

def save_gmm_models_pickle(gmm_models, save_path):
    """
    使用 pickle 保存整个 gmm_models 字典
    """
    with open(save_path, 'wb') as f:
        pickle.dump(gmm_models, f)
    print(f"✅ GMM 模型已保存到 {save_path}")

def load_gmm_models_pickle(load_path):
    """
    使用 pickle 加载 gmm_models 字典
    """
    with open(load_path, 'rb') as f:
        gmm_models = pickle.load(f)
    print(f"✅ 已加载 GMM 模型：{list(gmm_models.keys())}")
    return gmm_models

def load_voc_annotations(
    data_root: str,
    ann_file: str,
    img_subdir: str = 'JPEGImages',
    ann_subdir: str = 'Annotations',
    backend_args=None,
    show_progress=True,
):
    """
    从 VOC 格式数据集中加载所有 GT 框和标签，保证与 mmdet 训练管线一致。

    Args:
        data_root (str): 数据集根目录。
        ann_file (str): ImageSets/Main/*.txt 文件路径，相对 data_root。
        img_subdir (str): 图像子目录。
        ann_subdir (str): 标注子目录。
        resize (tuple | None): Resize 尺寸 (w, h)，若为 None 则不缩放。
        flip_prob (float): 随机翻转概率。设为 0 可禁用翻转。
        backend_args (dict | None): mmdet 的文件系统配置，如 Petrel。
        show_progress (bool): 是否显示 tqdm 进度条。

    Returns:
        all_bboxes (np.ndarray): 所有 GT 框 (N, 4)
        all_labels (np.ndarray): 对应的标签 (N,)
    """

    # ==== 构建 pipeline ====
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=backend_args),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='mmdet.PackDetInputs')
    ]

    # ==== 构建 dataset 配置 ====
    dataset_cfg = dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file=ann_file,
        ann_subdir=ann_subdir,
        img_subdir=img_subdir,
        data_prefix=dict(sub_data_root='', img=img_subdir),
        pipeline=train_pipeline,
        backend_args=backend_args,
    )

    dataset = DATASETS.build(dataset_cfg)
    class_names = dataset._metainfo["classes"]  # tuple，如 ('aeroplane', 'bicycle', ...)
    print(class_names)
    # ==== 遍历提取 bbox 与 label ====
    all_bboxes, all_labels = [], []
    iterator = tqdm(range(len(dataset)), desc='Loading GT from VOC') if show_progress else range(len(dataset))
    for i in iterator:
        sample = dataset[i]
        gt_instances = sample['data_samples'].gt_instances
        if len(gt_instances) == 0:
            continue
        all_bboxes.append(gt_instances.bboxes.numpy())
        all_labels.append(gt_instances.labels.numpy())

    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"✅ 共加载 {len(dataset)} 张图像，提取到 {len(all_bboxes)} 个 bbox。")
    print("bbox shape:", all_bboxes.shape)
    print("labels shape:", all_labels.shape)

    return all_bboxes, all_labels, class_names


def xyxy_to_xywh(boxes):
    """[x_min, y_min, x_max, y_max] -> [x_center, y_center, w, h]"""
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x_max - x_min
    h = y_max - y_min
    x = x_min + w / 2
    y = y_min + h / 2
    return np.stack([x, y, w, h], axis=1)


def xywh_to_xyxy(boxes):
    """[x_center, y_center, w, h] -> [x_min, y_min, x_max, y_max]"""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def fit_gmm_by_class_xywh(all_bboxes, all_labels, class_names, sub_list_info, n_components=3, random_state=42):
    """
    按类别训练 GMM，使用 xywh 形式拟合
    """
    class_bboxes = defaultdict(list)
    for bbox, label in zip(all_bboxes, all_labels):
        class_bboxes[class_names[label]].append(bbox)

    for sup_cls in sub_list_info:
        super_bboxes = []
        for sub_cls in sub_list_info[sup_cls]["leaves"]:
            super_bboxes.extend(class_bboxes[sub_cls])
        class_bboxes[sup_cls] = super_bboxes

    gmm_models = dict()
    for cls_name, bboxes in class_bboxes.items():
        print(f"Generate {cls_name} GMM........")
        X = np.array(bboxes)
        X_xywh = xyxy_to_xywh(X)  # 转换为 xywh
        if len(X_xywh) < n_components:
            n_comp = max(1, len(X_xywh))
        else:
            n_comp = n_components
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=random_state)
        gmm.fit(X_xywh)
        gmm_models[cls_name] = gmm

    return gmm_models, class_bboxes


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

if __name__ == '__main__':
    bboxes, labels, class_names = load_voc_annotations(
        data_root='data/OWOD/',
        ann_file='ImageSets/MOWODB/t1_train.txt',
        img_subdir='JPEGImages/MOWODB',
        ann_subdir='Annotations/MOWODB',
    )
    semantic_tree_path = "data/mowodb_tree_t1.json"
    # 加载语义树
    with open(semantic_tree_path, "r") as f:
        semantic_tree = json.load(f)
    sub_list_info = {}
    generate_sublist_full(semantic_tree, sub_list_info)

    # 假设你已经有 bboxes, labels, class_names
    gmm_models, class_bboxes = fit_gmm_by_class_xywh(bboxes, labels, class_names, sub_list_info, n_components=8)

    save_gmm_models_pickle(gmm_models, 'gmm_models.pkl')

    gmm_models_loaded = load_gmm_models_pickle('gmm_models.pkl')

    # 从类别 'car' 生成 5 个 bbox
    sampled_boxes = sample_gmm_xywh(gmm_models_loaded, 'car', n_samples=100, clip_box=(0, 0, 1333, 800))
    print("生成框：", sampled_boxes.tolist())