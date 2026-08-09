import json
import itertools
from openai import OpenAI
from tqdm import tqdm

# 类别列表
# VOC_CLASS_NAMES = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

# 类别列表
T2_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

# 生成所有两两组合

pair_list = list(itertools.product(T3_CLASS_NAMES, T2_CLASS_NAMES))

print("Total pairs:", len(pair_list))   # 190

client = OpenAI(
    api_key='sk-c6fd4844077b4797acdfa778b2d59829',  # 替换你的 key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ask_shared_attributes(cls1, cls2):
    """向模型询问：仅这两个类别共有的 5 个属性"""
    prompt = (
        f"We have two object classes: '{cls1}' and '{cls2}'.\n"
        "Your task is to output exactly 5 attributes that are shared exclusively by these two classes.\n"
        "These attributes must:\n"
        "1. Be common to both classes.\n"
        "2. NOT be common to most other everyday object classes.\n"
        "3. Cannot output empty \n"
        "Return ONLY a JSON array of 5 short attribute strings. For example:\n"
        '["attribute1", "attribute2", ..., "attribute5"]'
    )

    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are an expert in object attribute analysis."},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content


def main():
    print("=== Two-Class Shared Attribute Query ===\n")

    results = {}  # {"class1,class2": [att1, att2, ...]}

    for cls1, cls2 in tqdm(pair_list):

        pair_key = f"{cls1},{cls2}"
        print(f"\n🔍 Querying pair: {pair_key}")

        answer = ask_shared_attributes(cls1, cls2)
        print("➡️ Raw model answer:", answer)

        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            print(f"❌ Model returned non-JSON, saving raw text.")
            parsed = []


        results[pair_key] = parsed

    # 保存文件
    with open("pair_shared_attributes_t3_t1_two.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n✅ Saved to pair_shared_attributes_t3_t1_two.json")


if __name__ == "__main__":
    main()