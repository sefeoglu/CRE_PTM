"""Continuous instruction fine-tuning of Flan T5 model for Relation Extraction"""

import json

from datasets import load_dataset
from trl import SFTTrainer
from random import randrange

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from peft import  get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import evaluate
import nltk, torch
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")

from sklearn.metrics import precision_recall_fscore_support
from transformers import DataCollatorForSeq2Seq
# from huggingface_hub import HfFolder
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import concatenate_datasets


targets_labels = []
max_source_length = None
max_target_length = None

def set_dataset(dataset_id):
    # Load dataset

    dataset = load_dataset(dataset_id)
    # print(dataset['train'][0])
    return dataset


def set_tokenizer(model_id="google/flan-t5-large"):
    

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
def load_model(model_id="google/flan-t5-large"):
    
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id,  device_map="auto", quantization_config=bnb_config)
    return model




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
model = load_model(model_id="google/flan-t5-base")

metric = evaluate.load("rouge")

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
    
    return {k: round(v, 4) for k, v in result.items()}

def main(model_id, data_path, tasks_path, task_id):

    targets_labels = json.load(open(tasks_path))
    print(targets_labels)



    dataset = set_dataset(data_path)

    
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
    
    model = load_model(model_id)

  


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
    repository_id = f"{model_id.split('/')[1]}-{task_id}"

    training_args=TrainingArguments(
                output_dir=repository_id,
                num_train_epochs=1,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=1,
                optim="paged_adamw_32bit",
                save_steps=25,
                logging_steps=25,
                learning_rate=2e-2,
                fp16=False,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="constant",
                report_to="tensorboard"
            )
  

    trainer = SFTTrainer(
        model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    trainer.save_model()
    trainer.model.save_pretrained(repository_id)
    return trainer

if __name__ == "__main__":

    dataset_path = '/Users/sefika/phd_projects/CRE_PTM/data/tacred/continuous_prompt_fine_tuning/train/task1/'
    tasks_path = "/Users/sefika/phd_projects/CRE_PTM/data/tacred/train_tasks/relations/task1.json"
    model_id="google/flan-t5-large"
    task_id = "task1"
    trainer_model = main(model_id, dataset_path, tasks_path, task_id)
