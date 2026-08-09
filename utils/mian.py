import json
import networkx as nx
import torch

from transformers import AutoTokenizer, CLIPTextModelWithProjection


# --------------------------------------------
# 1) 文本编码函数（你提供的版本稍作优化）
# --------------------------------------------
model = './pretrain/clip-vit-base-patch32'
tokenizer = AutoTokenizer.from_pretrained(model)
model = CLIPTextModelWithProjection.from_pretrained(model)
device = 'cuda:0'
model.to(device)

def text_embedding(texts):
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    return txt_feats.reshape(-1, txt_feats.shape[-1]).to('cpu')

def deduplicate_by_clip_similarity(
    texts,
    sim_threshold=0.9
):
    """
    texts: List[str]
    sim_threshold: cosine similarity threshold
    return: deduplicated List[str]
    """

    assert len(texts) > 0

    # 1. 计算 CLIP 文本 embedding
    feats = text_embedding(texts)          # [N, D]
    feats = feats / feats.norm(dim=1, keepdim=True)

    keep_indices = []

    for i in range(len(texts)):
        keep = True
        for j in keep_indices:
            sim = torch.dot(feats[i], feats[j]).item()
            if sim >= sim_threshold:
                keep = False
                break
        if keep:
            keep_indices.append(i)

    return [texts[i] for i in keep_indices]

data = json.load(open("pairwise_shared_attributes_t3_t1.json"))

all_concepts = []

for k, v in data.items():
    all_concepts.extend(v)

all_concepts = list(set(all_concepts))

print(len(all_concepts))
all_concepts = deduplicate_by_clip_similarity(
    all_concepts,
    sim_threshold=0.90
)

json.dump(all_concepts, open("shared_attributes_t3_t1_list.json", "w"))
print(len(all_concepts))
