"""Build IPOW discriminative concepts for one incremental task.

For each ordered pair ``(positive_class, negative_class)``, Qwen returns one
visual attribute that the positive class has and the negative class lacks.
Results are resumable and saved in the ``pos_map_set``/``neg_map_set`` format
consumed by ``IPOWShared2FCBBoxHead``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import permutations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from share_concept_pipeline import (  # noqa: E402
    QwenClient,
    build_embeddings,
    load_dataset_classes,
    load_json,
    normalize_attribute,
    save_json,
)


LOGGER = logging.getLogger('ipow.dist_concept')


def task_pairs(current: list[str], previous: list[str]) -> list[tuple[str, str]]:
    """Return ordered pairs that add all current-task supervision."""
    pairs = list(permutations(current, 2))
    for new_class in current:
        for old_class in previous:
            pairs.append((new_class, old_class))
            pairs.append((old_class, new_class))
    return pairs


def query_discriminative_attributes(
        client: QwenClient, pairs: list[tuple[str, str]], path: Path
) -> dict[str, str]:
    results = load_json(path, {})
    skipped = 0
    for positive, negative in pairs:
        key = f'{positive} vs {negative}'
        if isinstance(results.get(key), str) and normalize_attribute(results[key]):
            skipped += 1
            continue
        prompt = (
            f'Between object classes "{positive}" and "{negative}", identify '
            f'ONE visual attribute that "{positive}" clearly has but '
            f'"{negative}" clearly does not have. The attribute must be a '
            'short, concrete, physically observable appearance property. Do '
            'not describe behavior, use, habitat, or context. Return only the '
            'attribute text, with no explanation.')
        try:
            results[key] = client.text(
                'You generate concise binary visual attributes.', prompt,
                context=f'discriminative:{key}')
        except RuntimeError:
            LOGGER.exception('failed discriminative query: %s', key)
            results[key] = ''
        save_json(path, results)
        LOGGER.info('discriminative query %d/%d: %s -> %s',
                    len(results), len(pairs), key, results[key])
    if skipped:
        LOGGER.info('resumed discriminative queries: skipped %d', skipped)
    return results


def merge_maps(
        raw: dict[str, str], previous_map: dict | None = None
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    positive: dict[str, set[str]] = {}
    negative: dict[str, set[str]] = {}
    if previous_map:
        positive.update({k: set(v) for k, v in previous_map.get('pos_map_set', {}).items()})
        negative.update({k: set(v) for k, v in previous_map.get('neg_map_set', {}).items()})
    for pair, value in raw.items():
        if not isinstance(value, str) or ' vs ' not in pair:
            continue
        attribute = normalize_attribute(value)
        if not attribute:
            continue
        positive_class, negative_class = pair.split(' vs ', 1)
        positive.setdefault(positive_class, set()).add(attribute)
        negative.setdefault(negative_class, set()).add(attribute)

    # The head requires one unambiguous label per class/attribute.
    conflicts = 0
    for class_name in set(positive) | set(negative):
        overlap = positive.get(class_name, set()) & negative.get(class_name, set())
        if overlap:
            conflicts += len(overlap)
            negative[class_name].difference_update(overlap)
    if conflicts:
        LOGGER.warning('removed %d contradictory class-attribute labels', conflicts)

    pos_map = {k: sorted(v) for k, v in positive.items() if v}
    neg_map = {k: sorted(v) for k, v in negative.items() if v}
    attributes = sorted({attribute for values in pos_map.values() for attribute in values} |
                        {attribute for values in neg_map.values() for attribute in values})
    return pos_map, neg_map, attributes


def main() -> None:
    parser = argparse.ArgumentParser(description='Build IPOW discriminative concepts')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--dataset', required=True, choices=['MOWODB', 'SOWODB', 'nuOWODB'])
    parser.add_argument('--task', required=True, type=int)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--previous-bank', default='')
    parser.add_argument('--clip-model', default='./pretrain/clip-vit-base-patch32')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--qwen-model', default='qwen3-max')
    parser.add_argument('--qwen-base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--retries', type=int, default=3)
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
    output_dir = Path(args.output_dir)
    seen, current, previous = load_dataset_classes(repo_root, args.dataset, args.task)
    pairs = task_pairs(current, previous)
    LOGGER.info('start discriminative pipeline: dataset=%s task=%d', args.dataset, args.task)
    LOGGER.info('seen=%d current=%d previous=%d pairs=%d',
                len(seen), len(current), len(previous), len(pairs))

    previous_path = Path(args.previous_bank) if args.previous_bank else None
    previous_map = {}
    previous_raw = {}
    if previous_path and previous_path.is_dir():
        previous_map = load_json(previous_path / 'concept_map.json', {})
        previous_raw = load_json(previous_path / 'pairwise_raw.json', {})
    elif previous_path:
        previous_map = load_json(previous_path, {})
        previous_raw = previous_map.get('pairwise_raw', {})

    client = QwenClient(args.qwen_model, args.qwen_base_url, args.retries)
    raw = dict(previous_raw)
    raw.update(query_discriminative_attributes(
        client, pairs, output_dir / 'pairwise_raw.json'))
    pos_map, neg_map, attributes = merge_maps(raw, previous_map)
    concept_map = {'pos_map_set': pos_map, 'neg_map_set': neg_map}
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / 'concept_map.json', concept_map)
    save_json(output_dir / 'attribute_bank.json', {
        'attributes': attributes, 'dataset': args.dataset, 'task': args.task})
    LOGGER.info('discriminative attributes: %d', len(attributes))
    output = output_dir / f'{args.dataset}_t{args.task}_dist_embeddings.pth'
    build_embeddings(attributes, concept_map, args.clip_model, args.device,
                     args.batch_size, output)


if __name__ == '__main__':
    main()
