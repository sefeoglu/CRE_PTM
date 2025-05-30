import torch
import json
import evaluate
import nltk, torch
import numpy as np
import trl

from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset


nltk.download("punkt")
metric = evaluate.load("rouge")



def load_tokenizer(model_id="mistralai/Mistral-7B-Instruct-v0.2"):
    """_summary_

    Args:
        model_id (str): id of the model on HF.

    Returns:
        tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return tokenizer

def get_targets(labels_file_path):
    """_summary_

    Args:
        labels_file_path (str): path of the file which gives the name of classes.

    Returns:
        list: the list of class/label names
    """

    targets = json.load(open(labels_file_path))
    # targets = [key for key in list(targets.keys())]

    return targets



def load_model(model_id, local):
    """Load the model from Hugging Face

    Args:
        model_id (str): the id of model on HF

    Returns:
        model: quantized pretrained language model
    """

    compute_dtype = getattr(torch, "float16")
    #quantization configs
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    if local:
      model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            quantization_config=quant_config,
            device_map={"": 0},
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map={"": 0}
            )

    #for single GPU
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


def load_dataset_from_hub(dataset_id):
    """Load dataset from the Hugging Face hub.

    Args:
        dataset_id (str): the id of dataset on HF

    Returns:
        val_dataset, train_dataset: the validation and train datasets
    """
    # Load dataset from the hub

    train_dataset = load_dataset(dataset_id, split="train")
    val_dataset = load_dataset(dataset_id, split="validation")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")


    return train_dataset, val_dataset


def formatting_prompts_func(example):
    """Create a list to store the formatted texts for each item in the example

    Args:
        example (list of dataset): one batch from dataset. each line might consist of prompt context and target_label
    Returns:
        formatted_texts: formated prompts
    """

    formatted_texts = []

    # Iterate through each example in the batch
    for text, raw_label in zip(example['prompt'], example['relation']):
        # Format each example as a prompt-response pair
        formatted_text = f"[INST] {text} [\INST] ### Answer:{raw_label}"
        formatted_texts.append(formatted_text)
    # Return the list of formatted texts
    return formatted_texts


def main(config, model_id, dataset_id, task_id, local, parameters):

    print("Fine tuning model: ", model_id, " on dataset: ", dataset_id)

    train_dataset, val_dataset = load_dataset_from_hub(dataset_id)

    # quantized pretrained model
    model = load_model(model_id, local)
    print("len:"+str(len(val_dataset)))
    lora_a = parameters['LORA']['lora_alpha']
    lora_d = parameters['LORA']['lora_dropout']
    lora_r = parameters['LORA']['r']
    model_name = config['MODEL']["model_id"]
    # apply LoRA configuration for CAUSAL LM, decode only models, such as Llama2-7B and Mistral-7B
    lora_config = LoraConfig(
        lora_alpha=lora_a,
        lora_dropout=lora_d,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    repository_id = f"{model_id}-{task_id}"

    model = get_peft_model(model, lora_config)
    tokenizer = load_tokenizer(model_name)
    epochs = parameters['PARAMETERS']['epochs']
    bs = parameters['PARAMETERS']['bs']
    lr = parameters['PARAMETERS']['lr']
    weight_decay = parameters['PARAMETERS']['weight_decay']
    #declare training arguments
    #please change it for more than one epoch. such as add val_loss for evaluation on epoch..
    training_args = TrainingArguments(
            do_eval=True,
            eval_strategy="epoch",
            output_dir=repository_id,
            num_train_epochs=epochs,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            save_steps=25,
            logging_steps=25,
            learning_rate=lr,
            weight_decay=weight_decay,
            fp16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            save_strategy="epoch",
            remove_unused_columns=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            lr_scheduler_type="cosine_with_restarts",
            lr_scheduler_kwargs = { "num_cycles": 1 },
            report_to="tensorboard",
        )

    response_template = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer( #based on RLHF
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            processing_class=tokenizer
        )

    trainer.train()
    torch.cuda.empty_cache()
    #save trainer
    trainer.save_model()
    trainer.model.save_pretrained(repository_id)
    trainer.tokenizer.save_pretrained(repository_id)
    merged_model = model.merge_and_unload()

    return merged_model, tokenizer, trainer
