import os
import re
import json
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
# Sample data

def extract_answers(data):
    results = []
    for entry in data:

        predict_text = entry["predict"]
        # Extract the sentence
        # print(predict_text)

        # Extract the relation type
        relation = predict_text.split("### Answer:")[1].strip().split('\n')[0].split(',')[0].split('.')[0].split(' ')[0]
        answer_match = re.search(r"Answer: (.*?)\n", predict_text)


        results.append({
                "sentence": predict_text,
                "relation": relation.replace('#','')
            })
        # break
    print(results)
    return results
def extact_answers(data, relation_types):
    for relation in relation_types:
        if relation in data:
            return relation
    return ''

def clean(data):
    task_relations = read_json("/Users/sefika/phd_projects/CRE_PTM/data/fewrel/all_fewrel_data.json")
    relation_types = [ item['relation'].replace(' ', "_") for item in task_relations]
    relation_types = list(set(relation_types))
    delimiter = "[/INST] ### Answer:"
    answers = []
    for i, item in enumerate(data):
        item['predict'] = item['predict'].replace("Answer: \n",delimiter)
        item['predict'] = item['predict'].replace("Answer:\n",'Answer:')
        answer = item['predict'].split(delimiter)[1]
        answer = answer.split("\n")[0].replace("#","").rstrip()
        answer = ' '.join(answer.split())
        answer = extact_answers(answer, relation_types)
        line = {"id":i, "predict":answer, "original":item['predict']}
        answers.append(line)

    return answers

def get_answers(folder_path):
    # Print results
    try:
        for i in range(1, 6):
            # llama_results/m_10/KMmeans_CRE_tacred
            folder = f'{folder_path}_{i}'
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                print(file)
                if file != ".ipynb_checkpoints" :
                    data =  read_json(file_path)

                    # # Extract answers
                    # extracted_answers = extract_answers(data)
                    extracted_answers = clean(data)
                    out_path = os.path.join(folder+'_extracted', file)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    write_json(extracted_answers, out_path)

    except Exception as e:
        print(f"file name: {file} ---> run {i}")
        print(e)

if __name__ == "__main__":
    ##TODO: Change the path ##
    folder_path = "/Users/sefika/phd_projects/CRE_PTM/resulting_metrics/results/fewrel/llama-results/fewrel/llama/m_10/KMmeans_CRE_tacred"
    get_answers(folder_path)
