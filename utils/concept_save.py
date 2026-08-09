import json
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection

# ======= 1. Load your attribute JSON =======

with open("/home/Newdisk1/lvxq/ITOW/pretrain/concept/share_concept_t3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to sorted reproducible list
attribute_list = sorted(list(data.keys()))

print(f"Total unique attributes: {len(attribute_list)}")


# ======= 3. Encode with CLIP =======
def text_embedding(texts):
    model = './pretrain/clip-vit-base-patch32'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = CLIPTextModelWithProjection.from_pretrained(model)
    device = 'cpu'
    model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    return txt_feats.reshape(-1, txt_feats.shape[-1]).to('cpu')


# Encode
print("\n🔁 Encoding attributes with CLIP...")
attribute_embeddings = text_embedding(attribute_list)

print("✔️ Encoding complete!")
print("Embedding shape:", attribute_embeddings.shape)

# ======= 4. Save results =======

save_data = {'embeddings':attribute_embeddings, 'concept_info': {'all_concept_list' :attribute_list, 'concept_map': data}}
torch.save(save_data, '/home/Newdisk1/lvxq/ITOW/pretrain/concept/share_concept_embeddings_t3.pth')

