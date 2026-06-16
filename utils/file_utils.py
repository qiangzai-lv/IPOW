import os
import random
import shutil


def sample_and_copy(src_dir, dst_dir, num_samples=500):
    # 支持的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 收集所有图片路径
    images = [
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.lower().endswith(exts)
    ]

    total = len(images)
    print(f"Found {total} images")

    if total == 0:
        print("No images found!")
        return

    # 如果不足500张，就全复制
    num_samples = min(num_samples, total)

    # 随机采样
    sampled = random.sample(images, num_samples)

    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)

    # 复制
    for img_path in sampled:
        file_name = os.path.basename(img_path)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy(img_path, dst_path)

    print(f"Copied {num_samples} images to {dst_dir}")


if __name__ == "__main__":
    src_dir = "/home/Newdisk1/lvxueqiang/ITOW/data/OWOD/JPEGImages/MOWODB"
    dst_dir = "/home/Newdisk1/lvxueqiang/ITOW/data/TestImages"

    sample_and_copy(src_dir, dst_dir, num_samples=50)