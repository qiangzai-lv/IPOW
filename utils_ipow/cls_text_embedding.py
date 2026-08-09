"""Generate CLIP text embeddings for IPOW known object classes."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import torch


LOGGER = logging.getLogger('ipow.cls_embedding')


def load_classes(repo_root: Path, dataset: str, task: int) -> list[str]:
    const_path = repo_root / 'mmdet' / 'datasets' / 'owodb_const.py'
    spec = importlib.util.spec_from_file_location('ipow_owod_const_cls', const_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load dataset constants from {const_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_ends = {
        'MOWODB': [0, 20, 40, 60, 80],
        'SOWODB': [0, 19, 40, 60, 80],
        'nuOWODB': [0, 10, 17, 23],
    }
    if dataset not in task_ends:
        raise ValueError(f'Unsupported dataset: {dataset}')
    if task < 1 or task >= len(task_ends[dataset]):
        raise ValueError(f'Invalid task {task} for {dataset}')
    end = task_ends[dataset][task]
    return list(module.VOC_COCO_CLASS_NAMES[dataset][:end])


def load_encoder(model_path: str, device: str):
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as exc:
        raise RuntimeError(
            'Cannot import Transformers CLIP encoder. Install a transformers '
            'version compatible with the server PyTorch version.') from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CLIPTextModelWithProjection.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def text_embedding(texts: list[str], tokenizer, model, device: str) -> torch.Tensor:
    """Use the original IPOW CLIP encoding and L2 normalization."""
    tokens = tokenizer(text=texts, return_tensors='pt', padding=True).to(device)
    outputs = model(**tokens)
    features = outputs.text_embeds
    features = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return features.reshape(-1, features.shape[-1]).cpu()


def save_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def generate_embeddings(classes: list[str], model_path: str, device: str) -> torch.Tensor:
    tokenizer, model = load_encoder(model_path, device)
    with torch.inference_mode():
        embeddings = text_embedding(classes, tokenizer, model, device)
    if embeddings.numel() == 0:
        raise ValueError('No classes were provided')
    LOGGER.info('encoded classes: %d', len(classes))
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate IPOW class embeddings')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--dataset', required=True, choices=['MOWODB', 'SOWODB', 'nuOWODB'])
    parser.add_argument('--task', required=True, type=int)
    parser.add_argument('--clip-model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='')
    parser.add_argument('--log-file', default='')
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=handlers, force=True)

    repo_root = Path(args.repo_root).resolve()
    classes = load_classes(repo_root, args.dataset, args.task)
    if args.output:
        output = Path(args.output)
    else:
        alias = {'MOWODB': 'mowod', 'SOWODB': 'sowod', 'nuOWODB': 'nuowod'}[args.dataset]
        output = Path('embeddings/cls_embeddings') / f'{alias}_t{args.task}_embeddings.pt'
    LOGGER.info('generating class embeddings: dataset=%s task=%d classes=%d',
                args.dataset, args.task, len(classes))
    LOGGER.info('model=%s device=%s output=%s', args.clip_model, args.device, output)
    embeddings = generate_embeddings(classes, args.clip_model, args.device)
    save_atomic(output, {'texts': classes, 'embeddings': embeddings})
    LOGGER.info('saved class embeddings: %s shape=%s', output, tuple(embeddings.shape))


if __name__ == '__main__':
    main()
