"""Continuous instruction fine-tuning of Flan T5 model for Relation Extraction"""

import json

from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import  get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from huggingface_hub import HfFolder
import evaluate
import nltk, torch
import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support
from transformers import DataCollatorForSeq2Seq

from datasets import concatenate_datasets

nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def set_dataset(dataset_id):
    # Load dataset

    dataset = load_dataset(dataset_id)

    print(len(dataset['validation']))
    return dataset


def set_tokenizer(model_id="google/flan-t5-base"):


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


tokenizer = set_tokenizer()
def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["prompt"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["relation"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# huggingface hub model id
def load_model(model_id="google/flan-t5-base", local=False):

    compute_dtype = getattr(torch, "float16")
    # load model from the hub
    # maxmem={i:f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB' for i in range()}
    # maxmem['cpu']='300GB'
    bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
    )

    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id,  device_map="auto", max_memory=maxmem, quantization_config=bnb_config)
    if local:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_id,  device_map="auto", quantization_config=bnb_config, local_files_only=True)
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_id,  device_map="auto", quantization_config=bnb_config)

    return model


def preprocess_logits_for_metrics(logits, labels):
  if isinstance(logits, tuple):
    logits = logits[0]

  return logits.argmax(dim=-1)
# helper function to postprocess text
def postprocess_text(labels, preds):
    preds = [pred.replace('\n','').split('Answer:')[-1].strip() for pred in preds]
    labels = [label.replace('\n','').split('Answer:')[-1].strip() for label in labels]
    #print(preds)
    #print(labels)
    return preds, labels



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    targets_labels = read_json(tasks_path)
    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
        # Some simple post-processing

    # model_predictions.extend(decoded_preds)
    grounds, preds = postprocess_text(decoded_labels,decoded_preds)
    p, r, f, _ = precision_recall_fscore_support(grounds, preds, labels=targets_labels, average='micro')
    acc = accuracy_score(grounds, preds)
    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]
    # Compute ROUGscores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(prediction_lens)

    result['f1'] = f
    result['recall'] =r
    result['precision']=p
    result['acc']=acc

    return {k: round(v, 4) for k, v in result.items()}

def main(model_id, data_path, tasks_path, task_id, local):

    targets_labels = read_json(tasks_path)
    print(targets_labels)

    lora_config = LoraConfig(
                  # the task to train for (sequence-to-sequence language modeling in this case)
                  task_type=TaskType.SEQ_2_SEQ_LM,
                  # the dimension of the low-rank matrices
                  r=4,
                  # the scaling factor for the low-rank matrices
                  lora_alpha=32,
                  # the dropout probability of the LoRA layers
                  lora_dropout=0.01,
                  target_modules=["k","q","v","o"]
                  )

    dataset = set_dataset(data_path)
    model = load_model(model_id, local)
    model = get_peft_model(model, lora_config)

    tokenizer = set_tokenizer(model_id="google/flan-t5-base")

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["validation"]]).map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True, remove_columns=["prompt", "relation"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["validation"]]).map(lambda x: tokenizer(x["relation"], truncation=True), batched=True, remove_columns=["prompt", "relation"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "relation"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")



    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )


    # Hugging Face repository id
    repository_id = f"{model_id}-{task_id}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=1e-3,
        num_train_epochs=5,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        lr_scheduler_type = "cosine_with_restarts",
        lr_scheduler_kwargs = { "num_cycles": 1 },
        remove_unused_columns=False
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,

    )


    # Start training
    trainer.train()
    trainer.save_model()
    trainer.model.save_pretrained(repository_id)
    merged_model = model.merge_and_unload()

    return merged_model, tokenizer, trainer


def get_prediction(model,tokenizer, prompt, length=250,stype='greedy'):

    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096,return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(inputs, max_new_tokens=length)

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return response

def Flan_T5_Trainer():
    for experiment_id in range(1, 6):

        print("Experiment: {0}".format(experiment_id))

        dataset_path = 'memory_based_cre/train/run_{0}/task1/'.format(experiment_id)
        tasks_path = "relations/run_{0}/task1.json".format(experiment_id)
        model_id="google/flan-t5-base"
        task_id = "task1"
        targets_labels = []
        max_source_length = None
        max_target_length = None
        tokenizer = set_tokenizer()

        lora_config = LoraConfig(
                        # the task to train for (sequence-to-sequence language modeling in this case)
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        # the dimension of the low-rank matrices
                        r=4,
                        # the scaling factor for the low-rank matrices
                        lora_alpha=32,
                        # the dropout probability of the LoRA layers
                        lora_dropout=0.01,
                        target_modules=["k","q","v","o"]
                        )
        metric = evaluate.load("rouge")
        model, tokenizer, trainer = main(model_id, dataset_path, tasks_path, task_id, False)

        model.save_pretrained("KMmeans_CRE_{0}/task_memory_20_1/".format(experiment_id), from_pt=True)
        ## evaluate model
        evaluate_model(experiment_id, task_id, model, tokenizer, current_task=True)

        train_data_path = dataset_path+"train_1.json"
        all_selected_samples = select_samples(model, tokenizer, 20, train_data_path, tasks_path)

        for k in range(2, 11):
            outpath_selected_samples = 'memory_based_cre/train/run_{0}/task_memory_{1}/train_2.json'.format(experiment_id,k)
            write_json(all_selected_samples, outpath_selected_samples)

        for i in range(1, 10):
            targets_labels = []
            max_source_length = None
            max_target_length = None
            tokenizer = set_tokenizer()
            metric = evaluate.load("rouge")

            dataset_path = 'memory_based_cre/train/run_{0}/task{1}/'.format(experiment_id,i+1)
            tasks_path = "relations/run_{0}/task{1}.json".format(experiment_id,i+1)

            base_model_id="KMmeans_CRE_{0}/task_memory_20_{1}/".format(experiment_id, i)
            task_id = "task{0}".format(i+1)

            print(base_model_id)

            model, tokenizer, trainer = main(base_model_id, dataset_path, tasks_path, task_id, True)
            #evaluate model
            evaluate_model(experiment_id, task_id, model, tokenizer, current_task=True)

            if i <9:
                train_data_path = dataset_path+"train_1.json"
                all_selected_samples = select_samples(model, tokenizer,20, train_data_path, tasks_path)
                for j in range(i+2, 11):
                    outpath_selected_samples = 'memory_based_cre/train/run_{0}/task_memory_{1}/train_{2}.json'.format(experiment_id, j, i+2)
                    write_json(all_selected_samples, outpath_selected_samples)
                else:
                    break

            ########################### Memory Train ######################################
            targets_labels = []
            max_source_length = None
            max_target_length = None
            tokenizer = set_tokenizer()
            metric = evaluate.load("rouge")

            dataset_path = 'memory_based_cre/train/run_{0}/task_memory_{1}/'.format(experiment_id,i+1)
            tasks_path = "relations/run_{0}/task{1}.json".format(experiment_id,i+1)

            base_model_id = "KMmeans_CRE_{0}/task_memory_20_{1}/".format(experiment_id, i+1)
            task_id = "task{0}".format(i+1)

            print(base_model_id)

            model, tokenizer, trainer = main(base_model_id, dataset_path, tasks_path, task_id, True)
            model.save_pretrained("KMmeans_CRE_{0}/task_memory_20_{1}/".format(experiment_id, i+1), from_pt=True)
            ### evaluate model
            evaluate_model(experiment_id, task_id, model, tokenizer, current_task=False)





