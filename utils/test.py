# 加载share att
import json

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = './pretrain/clip-vit-base-patch32'
tokenizer = AutoTokenizer.from_pretrained(model)
model = CLIPTextModelWithProjection.from_pretrained(model)
device = 'cuda:0'
model.to(device)
# --------------------------------------------
# 1) 文本编码函数（你提供的版本稍作优化）
# --------------------------------------------
def text_embedding(texts):
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    return txt_feats.reshape(-1, txt_feats.shape[-1]).to('cpu')


pre_data = json.load(open("/home/Newdisk1/lvxq/ITOW/pretrain/concept/share_concept_t3.json"))
pre_att = []
for k, v in pre_data.items():
    pre_att.append(k)

print(len(pre_att))

t2_data = json.load(open("/home/Newdisk1/lvxq/ITOW/pair_shared_attributes_t3_t1_two.json"))

with torch.no_grad():
    pre_att_feats = text_embedding(pre_att)   # [N_pre, D]

t2_att_map = {}

# ---------- 主循环 ----------
for k, att_list in tqdm(t2_data.items()):
    cls_pair = k.split(",")

    # 当前 pair 的 attribute embedding
    with torch.no_grad():
        att_feats = text_embedding(att_list)  # [N_att, D]

    # 相似度
    sim_matrix = att_feats @ pre_att_feats.T  # [N_att, N_pre]

    # 每个 att 与 pre_att 的最大相似度
    max_sim_per_att = sim_matrix.max(dim=1).values  # [N_att]

    # 找最不相似的 attribute
    min_idx = torch.argmin(max_sim_per_att).item()
    best_att = att_list[min_idx]
    best_score = max_sim_per_att[min_idx].item()

    print(f"{cls_pair}, 最不相似的属性:", best_att)
    print("对应最大相似度:", best_score)

    # 阈值过滤
    if best_score > 0.95:
        continue

    # 记录
    t2_att_map[best_att] = [cls_pair[0], cls_pair[1]]

    # ---------- 增量更新 ----------
    with torch.no_grad():
        new_feat = text_embedding([best_att])  # [1, D]

    pre_att.append(best_att)
    pre_att_feats = torch.cat([pre_att_feats, new_feat], dim=0)

json.dump(t2_att_map, open("t3_t1_att_map.json", "w"))
