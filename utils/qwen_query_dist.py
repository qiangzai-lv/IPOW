import json
from itertools import combinations, permutations, product
from openai import OpenAI
from tqdm import tqdm

# 类别列表
T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

client = OpenAI(
    api_key='sk-c6fd4844077b4797acdfa778b2d59829',  # 替换你的 key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ask_model(c1, c2):
    """向模型询问最具判别性的属性，仅针对 c1 拥有而 c2 不拥有"""
    prompt = (
        f"Between the following two object classes:\n"
        f"- {c1}\n"
        f"- {c2}\n\n"
        f"Identify ONE single visual attribute that {c1} clearly HAS but {c2} clearly DOES NOT HAVE.\n"
        f"This attribute must visually, and strictly differentiating.\n\n"
        "Respond ONLY with the attribute name (no extra words, no explanation, no JSON).\n"
    )

    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You generate concise binary visual attributes."},
            {"role": "user", "content": prompt},
        ]
    )

    return completion.choices[0].message.content.strip()


def main():
    print("=== Pairwise Discriminative Attribute Query ===\n")

    results = {}

    # 两两组合类别
    class_pairs = list(permutations(T3_CLASS_NAMES, 2))
    print(f"Total class pairs: {len(class_pairs)}\n")

    for c1, c2 in tqdm(class_pairs):
        pair_key = f"{c1} vs {c2}"
        print(f"🔍 Query: {pair_key}")

        answer = ask_model(c1, c2)
        print("➡️ Attribute:", answer, "\n")

        results[pair_key] = answer

    # 保存结果（JSON格式）
    with open("pairwise_best_attribute_t3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n🎉 Done! Saved to pairwise_best_attribute_t3.json")

if __name__ == "__main__":
    main()