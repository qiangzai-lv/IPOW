import os
import cv2
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm


def safe_find_text(node, tag, default=None):
    elem = node.find(tag)
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def visualize_voc(
    xml_dir,
    img_dir,
    save_dir=None,
    score_thr=0.0
):
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    xml_files.sort()

    for xml_name in tqdm(xml_files):
        xml_path = os.path.join(xml_dir, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = safe_find_text(root, "filename", xml_name.replace(".xml", ".jpg"))
        img_path = os.path.join(img_dir, filename)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        for obj in root.findall("object"):
            name = safe_find_text(obj, "name", "unknown")

            score = safe_find_text(obj, "score", None)
            score = float(score) if score is not None else None

            pseudo = safe_find_text(obj, "pseudo", "0")

            if score is not None and score < score_thr:
                continue

            bndbox = obj.find("bndbox")
            xmin = int(float(safe_find_text(bndbox, "xmin", 0)))
            ymin = int(float(safe_find_text(bndbox, "ymin", 0)))
            xmax = int(float(safe_find_text(bndbox, "xmax", 0)))
            ymax = int(float(safe_find_text(bndbox, "ymax", 0)))

            # 颜色区分：伪标注 vs GT
            if pseudo == "1":
                color = (0, 0, 255)   # 红色：pseudo
                tag = "P"
            else:
                color = (0, 255, 0)   # 绿色：GT
                tag = "G"

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

            label = name
            if score is not None:
                label += f" {score:.2f}"
            label += f" [{tag}]"

            cv2.putText(
                img,
                label,
                (xmin, max(ymin - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        if save_dir:
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    visualize_voc(
        'data/OWOD/Annotations_pseudo/MOWODB',
        'data/OWOD/JPEGImages/MOWODB',
        save_dir='visualize_voc',
        score_thr=0.5
    )