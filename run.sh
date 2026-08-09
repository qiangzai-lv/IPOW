PORT=29501 DATASET=MOWODB TASK=1 bash ./tools/dist_train.sh configs/ipow/ipow_owod_mowodb_t1.py 8 --amp

PORT=29502 DATASET=MOWODB TASK=1 bash ./tools/dist_train.sh configs/ipow/ipow_owod_mowodb_t1_120.py 8 --amp

python utils_ipow/gmm.py \
  --data-root /root/shared-nvme/data/OWOD \
  --dataset MOWODB \
  --task 4 \
  --n-components 8 \
  --output embeddings/gmm_rpn/mowodb_t4_class_gmm.pkl


export OPENAI_API_KEY="sk-c6fd4844077b4797acdfa778b2d59829"

python utils_ipow/share_concept_pipeline.py \
  --repo-root /root/shared-nvme/IPOW \
  --dataset MOWODB \
  --task 1 \
  --output-dir embeddings/concept_embeddings/mowodb_t1 \
  --clip-model /root/shared-nvme/data/pretrain/clip-vit-base-patch32 \
  --device cuda:0

python utils_ipow/dist_concept_pipeline.py \
  --repo-root /root/shared-nvme/IPOW \
  --dataset MOWODB \
  --previous-bank embeddings/concept_embeddings/mowodb_t1_dist \
  --task 2 \
  --output-dir embeddings/concept_embeddings/mowodb_t2_dist \
  --clip-model /root/shared-nvme/data/pretrain/clip-vit-base-patch32 \
  --device cuda:0

python utils_ipow/share_concept_pipeline.py \
  --repo-root /root/shared-nvme/IPOW \
  --dataset MOWODB \
  --task 2 \
  --previous-bank embeddings/concept_embeddings/mowodb_t1 \
  --output-dir embeddings/concept_embeddings/mowodb_t2 \
  --clip-model /root/shared-nvme/data/pretrain/clip-vit-base-patch32 \
  --device cuda:0


python utils_ipow/cls_text_embedding.py \
  --repo-root /root/shared-nvme/IPOW \
  --dataset MOWODB \
  --task 4 \
  --clip-model /root/shared-nvme/data/pretrain/clip-vit-base-patch32 \
  --device cuda:0
