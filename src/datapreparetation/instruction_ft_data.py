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
def get_prompt(sentence, head, tail):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """

    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Question: What is the relation type between tail and head entities?\n""" +\
                        """ Query Sentence: """ + str(sentence)+ """\n""" +\
                        """ head: """ + head + """. \n""" +\
                        """ tail: """ + tail + """. \n""" +\
                        """ output format: relation_type"""
    return template_zero_shot

def main(input_file, out_file):

    data = read_json(input_file)
    prompts = []

    for _, line in enumerate(data):
        prompt = get_prompt(line['sentence'], line['subject'], line['object'])
        prompts.append({'id':line['id'], 'prompt':prompt, 'relation':line['relation']})
    write_json(out_file, prompts)


if __name__ == "__main__":
    for i in range(1, 11):
        input_file_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/train_tasks/task{0}/dev.json".format(i)
        out_file_path="/Users/sefika/phd_projects/CRE_PTM/data/tacred/continuous_prompt_fine_tuning/train/task{0}/dev.json".format(i)
        main(input_file_path, out_file_path)