import os
import sys
import json
from sklearn.metrics import accuracy_score
def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    """ Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_results(predictions):
    clean_predictions = []

    for predict in predictions:
        responce = predict['predict']
        clean = responce.replace("\/",'')
        clean = clean.replace("[","").replace("]","").replace('"','').replace("'","").replace("org:","").replace("per:","")
        clean = clean.split(',')[0]
        clean_predictions.append(clean)

    return clean_predictions
                                 


def main(input_path, out_path, metric_path):
    results = []
    for k in range(1, 6):
        
        for t in range(1, 11):
            result = {}
            result['run'] = k
            result['task'] = t
            file_path = input_path+"result_task_{0}_{1}.json".format(k,t)
            data  =  read_json(file_path)
           
            # data = [item for item in data if  item['ground_truth'] in task]

            clean_predictions = clean_results(data)
            out_file = out_path + "result_task_{0}_{1}.json".format(k,t)

            write_json(clean_predictions,out_file)


            
            ground_truth_task = [line['ground_truth'].replace("per:","").replace("org:","") for line in data]
            print(clean_predictions[:4])
            print(ground_truth_task[:4])
 
     
            acc = accuracy_score( ground_truth_task, clean_predictions)
            result['acc'] = acc
            
            results.append(result)
    metric_file = metric_path
    write_json(results, metric_file)



        

if __name__ == "__main__":
    input_path = "/Users/sefika/phd_projects/CRE_PTM/src/test/results_memory_cl_fewrel/m_15/"
    out_path = "/Users/sefika/phd_projects/CRE_PTM/src/test/results_memory_cl_fewrel/m_15/"
    metric_path = "/Users/sefika/phd_projects/CRE_PTM/src/test/results_memory_cl_fewrel/m_15/t5_m_15_current_metrics.json"
    tasks_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/related_work_results/resluts/tacred_tasks.json"
    tasks = read_json(tasks_path)
    # main(input_path, out_path, metric_path)

    experiments = []
    for i in range(1,6):
        for k in range(1, 11):
            path = input_path+"run_{0}/task_task{1}_current_task_result.json".format(i,k)
            data = read_json(path)
            if type(data) == dict:
                data = [data]
            print(list(data[0].values())[0])
            row = {}
            row['run'] =  i
            row['task'] = k 
            row['acc'] = list(data[0].values())[0]
            experiments.append(row)
    write_json(experiments, metric_path)


