"""Build IPOW shared concepts for one incremental task.

The pipeline is resumable and produces the concept-map/embedding format used
by ``IPOWShared2FCBBoxHead``. API credentials are read from environment vars.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
from itertools import combinations, product
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger('ipow.share_concept')


def load_clip_text_encoder(model_path: str, device: str):
    """Load the project's original Transformers CLIP text encoder."""
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as exc:
        raise RuntimeError(
            'Cannot import Transformers CLIP text encoder. Check that the '
            'installed transformers version is compatible with the server '
            'PyTorch version (the project requires AutoTokenizer and '
            'CLIPTextModelWithProjection).') from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CLIPTextModelWithProjection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def text_embedding(texts: list[str], tokenizer, model, device):
    """Encode text with the original IPOW normalization and CPU output."""
    tokens = tokenizer(text=texts, return_tensors='pt', padding=True)
    tokens = tokens.to(device)
    text_outputs = model(**tokens)
    text_features = text_outputs.text_embeds
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.reshape(-1, text_features.shape[-1]).to('cpu')


def load_dataset_classes(repo_root: Path, dataset: str, task: int) -> tuple[list[str], list[str], list[str]]:
    """Return (seen classes, current classes, previous classes) for a task."""
    const_path = repo_root / 'mmdet' / 'datasets' / 'owodb_const.py'
    spec = importlib.util.spec_from_file_location('ipow_owod_const', const_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load dataset constants from {const_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    split_map = {
        'MOWODB': [0, 20, 40, 60, 80],
        'SOWODB': [0, 19, 40, 60, 80],
        'nuOWODB': [0, 10, 17, 23],
    }
    if dataset not in split_map:
        raise ValueError(f'Unsupported dataset: {dataset}')
    if task < 1 or task >= len(split_map[dataset]):
        raise ValueError(f'Invalid task {task} for {dataset}')
    all_classes = list(module.VOC_COCO_CLASS_NAMES[dataset])
    start, end = split_map[dataset][task - 1:task + 1]
    return all_classes[:end], all_classes[start:end], all_classes[:start]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with temp_path.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def normalize_attribute(value: str) -> str:
    value = re.sub(r'^[\s\-\d.)]+', '', str(value).strip().lower())
    value = re.sub(r'\s+', ' ', value)
    return value.strip(' .;:')


def parse_json_array(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.IGNORECASE)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError('Expected a JSON array')
    values = [normalize_attribute(item) for item in parsed if isinstance(item, str)]
    values = list(dict.fromkeys(item for item in values if item))
    if not values:
        raise ValueError('Empty attribute list')
    return values


class QwenClient:
    def __init__(self, model: str, base_url: str | None, retries: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install openai before running the query stage') from exc
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('Set OPENAI_API_KEY in the remote environment')
        kwargs = {'api_key': api_key}
        if base_url:
            kwargs['base_url'] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.retries = retries

    def json_array(self, system: str, prompt: str, context: str = '') -> list[str]:
        last_error = None
        for _ in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': prompt}],
                    temperature=0.3)
                raw = response.choices[0].message.content or ''
                LOGGER.info('model response [%s]: %s', context, raw.strip())
                return parse_json_array(raw)
            except Exception as exc:
                last_error = exc
                LOGGER.warning('Qwen request failed: %s', exc)
        raise RuntimeError(f'Qwen request failed after {self.retries} attempts: {last_error}')

    def text(self, system: str, prompt: str, context: str = '') -> str:
        last_error = None
        for _ in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': prompt}],
                    temperature=0.3)
                raw = response.choices[0].message.content or ''
                LOGGER.info('model response [%s]: %s', context, raw.strip())
                value = normalize_attribute(raw)
                if not value:
                    raise ValueError('Empty model response')
                return value
            except Exception as exc:
                last_error = exc
                LOGGER.warning('Qwen request failed: %s', exc)
        raise RuntimeError(f'Qwen request failed after {self.retries} attempts: {last_error}')


def query_pairwise_shared(client: QwenClient, pairs: list[tuple[str, str]], path: Path) -> dict:
    results = load_json(path, {})
    skipped = 0
    for first, second in pairs:
        key = f'{first} & {second}'
        if key in results and isinstance(results[key], list) and results[key]:
            skipped += 1
            continue
        prompt = (
            f'Between object classes "{first}" and "{second}", list at least '
            'five visual attributes that both classes share. Attributes must be '
            'short, concrete, physically observable appearance properties, not '
            'actions, uses, habitats, or contexts. Return only a JSON array of '
            'strings.')
        results[key] = client.json_array(
            'You generate concise shared visual attributes in JSON.', prompt,
            context=f'pairwise:{key}')
        save_json(path, results)
        LOGGER.info('pairwise query %d/%d: %s & %s -> %d attributes',
                    len(results), len(pairs), first, second, len(results[key]))
    if skipped:
        LOGGER.info('resumed pairwise queries: skipped %d completed pairs', skipped)
    return results


def candidate_attributes(pairwise: dict, existing: list[str]) -> list[str]:
    ordered = list(existing)
    seen = set(ordered)
    for values in pairwise.values():
        if not isinstance(values, list):
            continue
        for value in values:
            value = normalize_attribute(value)
            if value and value not in seen:
                ordered.append(value)
                seen.add(value)
    return ordered


def clip_deduplicate(attributes: list[str], model_path: str, device: str,
                     threshold: float, batch_size: int,
                     preserve_prefix: int = 0) -> list[str]:
    import torch

    tokenizer, model = load_clip_text_encoder(model_path, device)
    kept: list[str] = []
    kept_features = []
    with torch.inference_mode():
        for start in range(0, len(attributes), batch_size):
            batch = attributes[start:start + batch_size]
            features = text_embedding(batch, tokenizer, model, device)
            for offset, (attribute, feature) in enumerate(zip(batch, features)):
                is_preserved = start + offset < preserve_prefix
                if is_preserved or not kept_features:
                    keep = True
                else:
                    reference = torch.stack(kept_features)
                    keep = bool(torch.max(feature @ reference.T) < threshold)
                if keep:
                    kept.append(attribute)
                    kept_features.append(feature)
    return kept


def query_attribute_classes(client: QwenClient, attributes: list[str], classes: list[str], path: Path) -> dict[str, list[str]]:
    results = load_json(path, {})
    class_set = set(classes)
    skipped = 0
    for attribute in attributes:
        if attribute in results and isinstance(results[attribute], list):
            skipped += 1
            continue
        prompt = (
            f'Given these classes: {json.dumps(classes, ensure_ascii=False)}. '
            f'Which classes visibly possess the attribute "{attribute}"? '
            'Return only a JSON array containing class names from the given list. '
            'Return at least two classes or an empty array if the attribute is '
            'not shared.')
        try:
            values = client.json_array(
                'You assign visual attributes to a fixed list of object classes.', prompt,
                context=f'attribute:{attribute}')
        except RuntimeError:
            results[attribute] = []
        else:
            results[attribute] = list(dict.fromkeys(value for value in values if value in class_set))
        save_json(path, results)
        LOGGER.info('attribute query %d/%d: %s -> %d classes',
                    len(results), len(attributes), attribute, len(results[attribute]))
    if skipped:
        LOGGER.info('resumed attribute queries: skipped %d completed attributes', skipped)
    return results


def build_embeddings(attributes: list[str], concept_map: dict[str, list[str]],
                     model_path: str, device: str, batch_size: int, output: Path) -> None:
    import torch

    tokenizer, model = load_clip_text_encoder(model_path, device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(attributes), batch_size):
            chunks.append(text_embedding(attributes[start:start + batch_size],
                                         tokenizer, model, device))
    embeddings = torch.cat(chunks, dim=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'embeddings': embeddings,
                'concept_info': {'all_concept_list': attributes,
                                 'concept_map': concept_map}}, output)
    LOGGER.info('saved embeddings: %s, shape=%s', output, tuple(embeddings.shape))


def main() -> None:
    parser = argparse.ArgumentParser(description='Build IPOW shared concepts')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--dataset', required=True, choices=['MOWODB', 'SOWODB', 'nuOWODB'])
    parser.add_argument('--task', required=True, type=int)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--previous-bank', default='')
    parser.add_argument('--clip-model', default='./pretrain/clip-vit-base-patch32')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--clip-threshold', type=float, default=0.90)
    parser.add_argument('--qwen-model', default='qwen3-max')
    parser.add_argument('--qwen-base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--log-file', default='', help='Optional log file path')
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=handlers,
        force=True)

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    seen, current, previous = load_dataset_classes(repo_root, args.dataset, args.task)
    pairs = (list(combinations(current, 2)) if not previous else
             list(product(current, previous)) + list(combinations(current, 2)))
    LOGGER.info('start shared concept pipeline: dataset=%s task=%d', args.dataset, args.task)
    LOGGER.info('seen=%d current=%d previous=%d pairwise_pairs=%d',
                len(seen), len(current), len(previous), len(pairs))
    LOGGER.info('output_dir=%s clip_model=%s device=%s',
                output_dir, args.clip_model, args.device)
    client = QwenClient(args.qwen_model, args.qwen_base_url, args.retries)
    raw_path = output_dir / 'pairwise_raw.json'
    pairwise = query_pairwise_shared(client, pairs, raw_path)
    previous_path = Path(args.previous_bank) if args.previous_bank else None
    if previous_path and previous_path.is_dir():
        previous_bank = load_json(previous_path / 'attribute_bank.json', {})
        previous_map = load_json(previous_path / 'concept_map.json', {})
    else:
        previous_bank = load_json(previous_path, {}) if previous_path else {}
        previous_map = previous_bank.get('concept_map', {})
    old_attributes = list(previous_bank.get('attributes', []))
    LOGGER.info('loaded previous attributes: %d', len(old_attributes))
    attributes = candidate_attributes(pairwise, old_attributes)
    LOGGER.info('candidate attributes before CLIP deduplication: %d', len(attributes))
    attributes = clip_deduplicate(attributes, args.clip_model, args.device,
                                   args.clip_threshold, args.batch_size,
                                   preserve_prefix=len(old_attributes))
    LOGGER.info('attributes after CLIP deduplication: %d', len(attributes))
    save_json(output_dir / 'attribute_bank.json', {'attributes': attributes,
                                                   'dataset': args.dataset,
                                                   'task': args.task})
    new_concept_map = query_attribute_classes(
        client, attributes, seen, output_dir / 'attribute_to_classes.json')
    concept_map = dict(previous_map)
    for attribute, values in new_concept_map.items():
        if attribute in concept_map:
            values = list(dict.fromkeys(concept_map[attribute] + values))
        concept_map[attribute] = values
    concept_map = {attribute: values for attribute, values in concept_map.items()
                   if len(values) >= 2}
    attributes = [attribute for attribute in attributes if attribute in concept_map]
    LOGGER.info('valid shared concepts (at least two classes): %d', len(attributes))
    save_json(output_dir / 'attribute_bank.json', {'attributes': attributes,
                                                   'dataset': args.dataset,
                                                   'task': args.task})
    save_json(output_dir / 'concept_map.json', concept_map)
    build_embeddings(attributes, concept_map, args.clip_model, args.device,
                     args.batch_size,
                     output_dir / f'{args.dataset}_t{args.task}_share_embeddings.pth')


if __name__ == '__main__':
    main()
