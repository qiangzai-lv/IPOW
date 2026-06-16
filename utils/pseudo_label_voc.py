import argparse
import os
import warnings

import torch
from mmengine import Config
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector

warnings.filterwarnings('ignore')

import xml.etree.ElementTree as ET


class ObjectItem:
    def __init__(self):
        self.points = []
        self.possible_result_name = ""
        self.score = None


class Annotation:
    def __init__(self):
        self.filename = ""
        self.objects = []


def create_object_element(name, pose, truncated, difficult, bndbox, score):
    object_el = ET.Element('object')

    name_el = ET.SubElement(object_el, 'name')
    name_el.text = name

    pose_el = ET.SubElement(object_el, 'pose')
    pose_el.text = pose

    truncated_el = ET.SubElement(object_el, 'truncated')
    truncated_el.text = str(truncated)

    difficult_el = ET.SubElement(object_el, 'difficult')
    difficult_el.text = str(difficult)

    score_el = ET.SubElement(object_el, 'score')
    score_el.text = str(score)

    pseudo_el = ET.SubElement(object_el, 'pseudo')
    pseudo_el.text = str(1)

    bndbox_el = ET.SubElement(object_el, 'bndbox')
    xmin_el = ET.SubElement(bndbox_el, 'xmin')
    xmin_el.text = str(int(bndbox[0]))
    ymin_el = ET.SubElement(bndbox_el, 'ymin')
    ymin_el.text = str(int(bndbox[1]))
    xmax_el = ET.SubElement(bndbox_el, 'xmax')
    xmax_el.text = str(int(bndbox[2]))
    ymax_el = ET.SubElement(bndbox_el, 'ymax')
    ymax_el.text = str(int(bndbox[3]))

    return object_el


def parse_xml_for_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = Annotation()
    annotation.filename = root.find('filename').text
    for obj in root.findall('object'):
        obj_item = ObjectItem()
        bandbox = obj.find('bndbox')
        xmin = int(bandbox.find('xmin').text is None and 0 or bandbox.find('xmin').text)
        ymin = int(bandbox.find('ymin').text is None and 0 or bandbox.find('ymin').text)
        xmax = int(bandbox.find('xmax').text is None and 0 or bandbox.find('xmax').text)
        ymax = int(bandbox.find('ymax').text is None and 0 or bandbox.find('ymax').text)
        obj_item.points.append(xmin)
        obj_item.points.append(ymin)
        obj_item.points.append(xmax)
        obj_item.points.append(ymax)
        score = obj.find('score')
        obj_item.score = float(score.text) if score is not None else ''
        cls_name = obj.find('name').text
        obj_item.cls_name = cls_name
        annotation.objects.append(obj_item)
    return annotation


def add_objects_to_xml(input_xml_path, out_xml_path, boxes, types, scores, iou_thr):
    # 解析XML文件
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    voc_label = parse_xml_for_voc(input_xml_path)
    bboxs = [obj.points for obj in voc_label.objects]
    for box, cls, score in zip(boxes, types, scores):
        max_iou = calculate_max_iou(box, bboxs)
        if max_iou < iou_thr:
            # 创建object元素
            object_el = create_object_element(cls, 'Unspecified', 0, 0, box, score)
            # 将object元素添加到XML树中
            root.append(object_el)

    # 将修改后的XML保存到文件
    tree.write(out_xml_path)


def calculate_iou(box1, box2):
    # 计算两个边界框的交集部分
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 如果没有交集则返回0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou


def calculate_max_iou(bbox, bboxs):
    max_iou = 0
    for b in bboxs:
        iou = calculate_iou(bbox, b)
        if iou > max_iou:
            max_iou = iou
    return max_iou


def load_imageset(imageset_path):
    """
    读取 ImageSets txt 文件，返回 image id 的 set
    """
    with open(imageset_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]
    return set(image_ids)


class Output:
    def __init__(self, xyxy: list, scores: list, cls: list, path):
        self.xyxy = xyxy
        self.scores = scores
        self.cls = cls
        self.path = path


class MmdetModel:
    def __init__(self, cfg_path, pt_path, skip_scores=0.5) -> None:
        self.cfg_path = cfg_path
        self.pt_path = pt_path
        self.skip_scores = skip_scores
        known_text_embeddings_path = Config.fromfile(cfg_path)["owod_cfg"]['known_text_embeddings_path']
        known_classes = torch.load(known_text_embeddings_path)['texts']
        # COCO类别映射
        self.class_names = known_classes
        self.model = init_detector(self.cfg_path, self.pt_path)

    def predict(self, img_path):
        result = inference_detector(self.model, img_path)
        # print(result.pred_instances)
        labels = result.pred_instances.labels
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores
        ins = scores > self.skip_scores
        bboxes = bboxes[ins, :]
        labels = labels[ins]
        scores = scores[ins]
        # filter unknown
        known_index = labels < len(self.class_names)
        bboxes = bboxes[known_index]
        labels = labels[known_index]
        scores = scores[known_index]

        return Output(
            bboxes.tolist(),
            scores.tolist(),
            [self.class_names[cls_idx] for cls_idx in labels],
            img_path
        )


def main(imageset_path, input_images_path, input_labels_path, out_labels_path, cfg, pt, skip_scores, iou_thr,
         num_chunks=1,
         chunk_idx=0):
    assert 0 <= chunk_idx < num_chunks, "Invalid chunk_idx"

    model = MmdetModel(cfg, pt, skip_scores=skip_scores)

    if not os.path.exists(out_labels_path):
        os.makedirs(out_labels_path)

    valid_image_ids = load_imageset(imageset_path)
    print(f'Loaded {len(valid_image_ids)} images from {imageset_path}')

    valid_input_labels_paths = []
    for i in tqdm(os.listdir(input_labels_path)):
        image_name = i.replace('.xml', '')
        if image_name in valid_image_ids:
            valid_input_labels_paths.append(i)

    valid_input_labels_paths.sort()
    total = len(valid_input_labels_paths)

    start_idx = total * chunk_idx // num_chunks
    end_idx = total * (chunk_idx + 1) // num_chunks

    print(f"[Chunk {chunk_idx}/{num_chunks}] "
          f"Processing [{start_idx}:{end_idx}) / {total}")

    valid_input_labels_paths = valid_input_labels_paths[start_idx:end_idx]

    for i in tqdm(valid_input_labels_paths):
        if not i.endswith('.xml'):
            continue
        image_name = i.replace('.xml', '')
        img_path = os.path.join(input_images_path, image_name + '.jpg')
        if not os.path.exists(img_path):
            continue
        result = model.predict(img_path)
        input_xml_path = os.path.join(input_labels_path, i)
        output_xml_path = os.path.join(out_labels_path, i)
        add_objects_to_xml(input_xml_path, output_xml_path,
                           result.xyxy, result.cls, result.scores, iou_thr)


def parse_args():
    parser = argparse.ArgumentParser(description="Add predicted objects to VOC XML annotations with IoU filtering")
    parser.add_argument("--input_images", type=str, default='data/OWOD/JPEGImages/SOWODB', help="Path to input images")
    parser.add_argument("--mmdet_cfg", type=str, help="Path to input images")
    parser.add_argument("--mmdet_pt", type=str, help="Path to input images")
    parser.add_argument("--imageset_path", type=str, help="Path to input images")
    parser.add_argument("--skip_scores", type=float, default=0.9, help="Score threshold for predictions")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for filtering predictions")
    # ⭐ 新增：分块伪标注参数
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="total number of chunks")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="current chunk index [0, num_chunks)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    input_labels = 'data/OWOD/Annotations/SOWODB'
    output_labels_path = 'data/OWOD/Annotations_pseudo/SOWODB'
    if not os.path.exists(output_labels_path):
        os.makedirs(output_labels_path)
    main(args.imageset_path, args.input_images, input_labels, output_labels_path, args.mmdet_cfg, args.mmdet_pt,
         args.skip_scores, args.iou_thr, args.num_chunks, args.chunk_idx)
