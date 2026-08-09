import torch
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)


def text_embedding(texts):
    model = './pretrain/clip-vit-base-patch32'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = CLIPTextModelWithProjection.from_pretrained(model)
    device = 'cuda:0'
    model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    return txt_feats.reshape(-1, txt_feats.shape[-1]).to('cpu')


if __name__ == "__main__":
    # 示例文本
    texts = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "microwave", "oven", "toaster", "sink", "refrigerator",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"

    ]

    # 提取文本 embedding
    embeddings = text_embedding(texts)

    # 保存为 .pt 文件
    torch.save({"texts": texts, "embeddings": embeddings}, "pretrain/embeddings/mowod_t3_embeddings.pt")
    print(f"✅ 保存完成: text_embeddings.pt {embeddings.shape}")
