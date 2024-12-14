import sys
import os
import json
def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def prepare_data_re(data):
    dataset = []
    # print(data.keys())
    relation_id = read_json("/Users/sefika/phd_projects/CRE_PTM/data/fewrel/data/pid2name.json")
    # print(len(relation_id))
    for i, relation in enumerate(data.keys()):
        sentences = data[relation]
        # print(relation_id[relation][0])
        for line in sentences:
            # print(line)
            tokens = line['tokens']
            print(line['h'][2])
            print(tokens[line['h'][2][0][0]])

            sentence = " ".join([t for t in tokens])
            head = " ".join([ tokens[id] for id in line['h'][2][0]])
            tail = " ".join([tokens[id] for id in line['t'][2][0]])
            raw_data = {
                "sentence":sentence,
                "tokens":tokens,
                "subject":head,
                "object":tail,
                "relation": relation_id[relation][0],
                "relation_PID":relation
            }
            dataset.append(raw_data)

    return dataset

def main(file_path, out_path):
    data =  read_json(file_path)
    dataset = prepare_data_re(data)
    val_file_path = "/Users/sefika/phd_projects/CRE_PTM/data/fewrel/data/val_wiki.json"
    val_data =  read_json(val_file_path)
    val_dataset = prepare_data_re(val_data)
    dataset.extend(val_dataset)
    write_json(out_path, dataset)

    
if __name__ =="__main__":
    file_path = "/Users/sefika/phd_projects/CRE_PTM/data/fewrel/data/train_wiki.json"
    out_path = "/Users/sefika/phd_projects/CRE_PTM/data/fewrel/final/train_wiki.json"
    main(file_path, out_path)