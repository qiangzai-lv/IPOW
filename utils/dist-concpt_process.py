import json

base_map_set = json.load(open("up_map_set_up_t3.json"))

t2_data = json.load(open("pairwise_best_attribute_t3_t1.json", "r", encoding="utf-8"))

pos_map_set = base_map_set['pos_map_set']
neg_map_set = base_map_set['neg_map_set']

for pair, att in t2_data.items():
    pair = pair.split("vs")
    t_cls = pair[0].strip()
    f_cls = pair[1].strip()
    att = att.lower()
    if t_cls not in pos_map_set:
        pos_map_set[t_cls] = []
    else:
        pos_map_set[t_cls].append(att)

    if f_cls not in neg_map_set:
        neg_map_set[f_cls] = []
    else:
        neg_map_set[f_cls].append(att)

dump_pos_map_set = {k: list(set(v)) for k, v in pos_map_set.items()}
dump_neg_map_set = {k: list(set(v)) for k, v in neg_map_set.items()}

json.dump({'pos_map_set': dump_pos_map_set, 'neg_map_set': dump_neg_map_set},
          open("up_map_set_up_t3_t1.json", "w", encoding="utf-8"))
