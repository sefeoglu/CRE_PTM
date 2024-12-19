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

def get_answers(folder_path):
    # Print results
    for i in range(1, 6):
        # llama_results/m_10/KMmeans_CRE_tacred
        folder = f'{folder_path}_{i}'
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            print(file)
            if file != ".ipynb_checkpoints"  :
                data =  read_json(file_path)

                # # Extract answers
                extracted_answers = extract_answers(data)
                out_path = os.path.join(folder+'_extracted', file)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                write_json(extracted_answers, out_path)
if __name__ == "__main__":
    ##TODO: Change the path ##
    folder_path = ""
    get_answers(folder_path)
