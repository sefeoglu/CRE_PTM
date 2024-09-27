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


def main(input_file, out_file, out_file_patterns):

    dataset = []
    patterns = []

    data = read_json(input_file)

    for i, line in enumerate(data):
        entity_pattern = {
                "id":line['id'],
                "subject_entity":{"entity": line['subject'],"type":line['subject_type']},
                "object_entity":{"entity": line['object'],"type":line['object_type']},
                "relation":line['relation']
            }
        pattern = {
            "id":line['id'],
            "subject_type":line['subject_type'],
            "object_type":line['object_type'],
            "relation":line['relation']
        }
        dataset.append(entity_pattern)
        patterns.append(pattern)
    #save dataset
    write_json(out_file, dataset)
    write_json(out_file_patterns, patterns)


if __name__ == "__main__":

    input_file = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/final/test.json"
    out_file = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/entity_pairs/test.json"
    out_file_patterns = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/data/patterns/test.json"
    main(input_file, out_file, out_file_patterns)
