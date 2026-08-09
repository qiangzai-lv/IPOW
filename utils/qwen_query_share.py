import itertools
import json
from itertools import combinations, product
from openai import OpenAI
from tqdm import tqdm

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

def ask_shared_attributes(c1, c2):
    prompt = (
        f"Between the following two object classes:\n"
        f"- {c1}\n"
        f"- {c2}\n\n"
        "List at least 5 shared visual attributes that BOTH categories have.\n"
        "These attributes should ideally be features that are **shared mainly between these two classes, and are uncommon or absent in most other classes**.\n"
        "Attributes must describe **visually observable physical appearance only** "
        "Do NOT include behavior, purpose, usage, habitat, context, or functionality.\n\n"
        "Formatting rules:\n"
        "- Output must be a **pure JSON array of short attribute strings**.\n"
        "- Each attribute should be concise and concrete.\n"
        "- No explanation, no reasoning, no ranking, no text outside the JSON.\n\n"
        "Now respond:"
    )

    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You generate concise shared visual attribute lists in JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3
    )

    raw = completion.choices[0].message.content.strip()

    # 尝试解析，失败就返回空
    try:
        print(f'{c1}&{c2}', json.loads(raw))
        return json.loads(raw)
    except:
        print(f"error: {raw}")
        return [raw]  # fallback


def main():
    print("=== Pairwise Shared Visual Attribute Query ===\n")

    results = {}
    class_pairs = list(itertools.combinations(T3_CLASS_NAMES, 2))
    print(f"Total class pairs: {len(class_pairs)}\n")

    for c1, c2 in tqdm(class_pairs):
        pair_key = f"{c1} & {c2}"
        answer = ask_shared_attributes(c1, c2)
        results[pair_key] = answer

    # 保存
    with open("pairwise_shared_attributes_t3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n🎉 Done! Saved to pairwise_shared_attributes_t3.json")


if __name__ == "__main__":
    main()