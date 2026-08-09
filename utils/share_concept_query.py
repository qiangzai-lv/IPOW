import json

from openai import OpenAI
from tqdm import tqdm

# 类别列表
T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]


# 属性列表 —— 修改这里即可
att_list = json.load(open("shared_attributes_t3_list.json", "r", encoding="utf-8"))

print('att_list:', len(att_list))

client = OpenAI(
    api_key='sk-c6fd4844077b4797acdfa778b2d59829',  # 替换你的 key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ask_model(attribute):
    """向模型提问：哪些类具有该属性"""
    prompt = (
        "You are given the following object classes:\n"
        f"{', '.join(T3_CLASS_NAMES)}.\n\n"
        f"Question: Which of these classes possess the attribute '{attribute}'?\n\n"
        "Rules:\n"
        "1. Only select from the given classes.\n"
        "2. Respond ONLY with a JSON array of class names.\n\n"
        'Example:\n["cat", "dog"]'
    )

    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a category attribute assigner."},
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content


def main():
    print("=== Attribute-to-Class Query Script ===\n")

    results = {}  # 用来保存最终 JSON 数据

    for att in tqdm(att_list):
        print(f"🔍 Querying attribute: {att}")
        answer = ask_model(att)
        print("➡️ Model answer:", answer, "\n")

        # 尝试将模型回答解析为 JSON
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            print(f"❌ Warning: Model returned non-JSON for attribute '{att}'. Saving raw text.")
            parsed = answer

        classes = []
        for cls in parsed:
            if cls in T3_CLASS_NAMES:
                classes.append(cls)
            else:
                print(cls, 'not in T2_CLASS_NAMES')
        results[att] = classes

    # 保存到文件
    with open("share_att_results_t3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("✅ All results saved to results.json")


if __name__ == "__main__":
    main()