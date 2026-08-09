import json

# 加载share att
import json

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = './pretrain/clip-vit-base-patch32'
tokenizer = AutoTokenizer.from_pretrained(model)
model = CLIPTextModelWithProjection.from_pretrained(model)
device = 'cpu'
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

threshold = 0.93

mu_data = json.load(open("/home/Newdisk1/lvxq/ITOW/pretrain/concept/share_att_results_mu_t3.json"))

# 删除属性值为1
mu_atts = []
for k, v in mu_data.items():
    if len(v) > 1:
        mu_atts.append(k)

pre_att_feats = text_embedding(pre_att)   # [N_pre, D]
mu_att_feats  = text_embedding(mu_atts)   # [N_mu, D]

saved_mu_atts = []
saved_mu_feats = []

# 当前参考特征（动态增长）
ref_feats = pre_att_feats.clone()  # [N_ref, D]

for i, att in enumerate(mu_atts):
    att_feat = mu_att_feats[i].unsqueeze(0)  # [1, D]

    # 与当前 ref 的最大相似度
    sim = (att_feat @ ref_feats.T).max().item()

    if sim < threshold:
        saved_mu_atts.append(att)
        saved_mu_feats.append(att_feat)

        # 动态加入 reference
        ref_feats = torch.cat([ref_feats, att_feat], dim=0)

print("最终保存 mu_atts 数量:", len(saved_mu_atts))

save_data = {}
for k in saved_mu_atts:
    save_data[k] = mu_data[k]


with open(
    "/home/Newdisk1/lvxq/ITOW/pretrain/concept/filtered_mu_atts_greedy_t3.json", "w"
) as f:
    json.dump(save_data, f, indent=2)