

import evaluate
import nltk
from datetime import datetime
from memory.kmeans_sampleselection import select_samples
from cre_t5 import set_tokenizer, main

nltk.download("punkt")
metric = evaluate.load("rouge")
from evaluation import evaluate_model, write_json

def trainer(config, memory_size=10):
    logs = ""

    m = memory_size
    model_id = config['MODEL']["model_id"]
    base_model_name = config['MODEL']["base_model_id"]
    dataset = config['DATASET']['dataset_name']
    test_results_folder = config['TEST']['test_results_folder']
    test_dataset_folder = config['TEST']['test_dataset_folder']
    for experiment_id in range(1,6):

        print(f"Experiment: {experiment_id}")

        dataset_path = f"{dataset}/train/run_{experiment_id}/task1/"
        tasks_path = f"{dataset}/relations/run_{experiment_id}/task1.json"
        model_id="google/flan-t5-base"
        task_id = "task1"
        tokenizer = set_tokenizer()

        metric = evaluate.load("rouge")
        start_time = datetime.now()
        model, tokenizer, trainer = main(config, model_id, dataset_path, tasks_path, task_id, False)
        end_time = datetime.now()
        train_time = 'Base Train. Experiment Id: {0}. Task Id: {1}. Duration: {2} \n'.format(experiment_id, task_id, end_time - start_time)
        logs += train_time

        model.save_pretrained(f"{base_model_name}_{experiment_id}/task_memory_{memory_size}_1/", from_pt=True)
        ## evaluate model
        evaluate_model(test_results_folder, test_dataset_folder, experiment_id, task_id, model, tokenizer, model_id, current_task=True)
        
        if memory_size >0:
            train_data_path = dataset_path + "train_1.json"
            all_selected_samples = select_samples(model, tokenizer, memory_size, train_data_path, tasks_path)

            for k in range(2, 11):
                outpath_selected_samples = f"{dataset}/train/run_{experiment_id}/task_memory_{k}/train_2.json"
                write_json(all_selected_samples, outpath_selected_samples)

        for i in range(1, 10):
            targets_labels = []
            max_source_length = None
            max_target_length = None
            tokenizer = set_tokenizer()
            metric = evaluate.load("rouge")

            dataset_path = f"{dataset}/train/run_{experiment_id}/task{i+1}/"
            tasks_path = f"{dataset}/relations/run_{experiment_id}/task{i+1}.json"

            base_model_id = f"{base_model_name}_{experiment_id}/task_memory_{memory_size}_{i}/"
            task_id = "task{0}".format(i+1)

            print(base_model_id)
            start_time = datetime.now()
            model, tokenizer, trainer = main(base_model_id, dataset_path, tasks_path, task_id, True)
            end_time = datetime.now()
            train_time = 'Base Train. Experiment Id: {0}. Task Id: {1}. Duration: {2} \n'.format(experiment_id, task_id, end_time - start_time)
            logs += train_time
            model.save_pretrained(f"{base_model_name}_{experiment_id}/task_memory_{memory_size}_{i+1}/", from_pt=True)
            #evaluate model
            
            evaluate_model(test_results_folder, test_dataset_folder, experiment_id, task_id, model, tokenizer, model_id, current_task=True)
            write_json(logs, f"{experiment_id}/_{experiment_id}/logs.txt")
            if memory_size > 0:
                if i <9:
                    train_data_path = dataset_path+"train_1.json"
                    all_selected_samples = select_samples(model, tokenizer,m, train_data_path, tasks_path)
                    for j in range(i+2, 11):
                        outpath_selected_samples = f'{dataset}/train/run_{experiment_id}/task_memory_{j}/train_{i+2}.json'
                        write_json(all_selected_samples, outpath_selected_samples)

                ########################### Memory Train ######################################
  
                tokenizer = set_tokenizer()
                metric = evaluate.load("rouge")

                dataset_path = f"{dataset}/train/run_{experiment_id}/task_memory_{i+1}/"
                tasks_path = f"{dataset}/relations/run_{experiment_id}/task{i+1}.json"

                base_model_id = f"{base_model_name}_{experiment_id}/task_memory_{memory_size}_{i+1}/"
                task_id = "task{0}".format(i+1)

                print(base_model_id)
                start_time = datetime.now()
                model, tokenizer, trainer = main(base_model_id, dataset_path, tasks_path, task_id, True)
                end_time = datetime.now()

                train_time = 'Memory Train. Experiment Id: {0}. Task Id: {1}. Duration: {2} \n'.format(experiment_id, task_id, end_time - start_time)
                logs += train_time
                model.save_pretrained(f"{base_model_name}_{experiment_id}/task_memory_{memory_size}_{i+1}/", from_pt=True)
                ### evaluate model
              
                
                evaluate_model(test_results_folder, test_dataset_folder, experiment_id, i+1, model, tokenizer, model_id, current_task=False)
                write_json(logs, f"{base_model_name}_{experiment_id}/logs.txt")
