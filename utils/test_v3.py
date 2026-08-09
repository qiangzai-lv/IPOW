import json



load = json.load(open('/home/Newdisk1/lvxq/ITOW/pretrain/concept/dist_attribute_name_t3.json'))

neg_map_set = load['neg_map_set']
pos_map_set = load['pos_map_set']

for k in pos_map_set:
    pos_stt = neg_map_set[k]
    neg_stt = pos_map_set[k]
    inter_stt = list(set(pos_stt) & set(neg_stt))
    if len(inter_stt) > 0:
        print(k, inter_stt)

