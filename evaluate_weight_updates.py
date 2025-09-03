# from utils import load_model, load_tokenizer, write_csv, read_json, write_json, load_csv
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN
from datasets import load_dataset, Dataset

import os
import torch
from typing import List, Dict, Literal
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model(model, tokenizer, tokenized_dataset, save_dir, seed=42):
    # n_gpus = torch.cuda.device_count()
    # if n_gpus > 1:
    #     print(f"Using {n_gpus} GPUs via DataParallel.")
    #     model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))

    model = model.to(device)
    model.train()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    batch_size = 8  # adjust as needed
    train_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    epochs = 2

    weight_change_norms_per_epoch = {}
    perplexities = []

    # Compute initial perplexity
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Perplexity init"):
            # input_ids = batch['input_ids'].to(device)
            input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']]).to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in attention_mask]).to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    perplexities.append(perplexity)
    print(f"Initial perplexity: {perplexity:.4f}")
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        weight_change_norms = []
        prev_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']]).to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in attention_mask]).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_state_dict = model.state_dict()
            norm = 0.0
            for k in prev_state_dict:
                norm += torch.norm(current_state_dict[k] - prev_state_dict[k], p=2).item()
            weight_change_norms.append(norm)
            prev_state_dict = {k: v.clone().detach() for k, v in current_state_dict.items()}

        weight_change_norms_per_epoch[epoch] = weight_change_norms
        print(f"Weight change norms for epoch {epoch+1}: {np.mean(weight_change_norms)}")

        # Compute perplexity on the finetuning dataset
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in tqdm(train_loader, desc=f"Perplexity Epoch {epoch+1}"):
                # input_ids = batch['input_ids'].to(device)
                input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']]).to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in attention_mask]).to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        perplexities.append(perplexity)
        print(f"Perplexity after epoch {epoch+1}: {perplexity:.4f}")
        model.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return weight_change_norms_per_epoch, perplexities


def load_model(model_name_base, step):
    model_name = f"EleutherAI/{model_name_base}"
    revision = f"step{step}"
    cache_dir = f"./{model_name_base}/step{step}"
    print('cache_dir:', cache_dir)

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )

    print(f"Loaded model {model_name} at step {step}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )

    print(f"Loaded tokenizer for {model_name} at step {step}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_data(tokenizer, files, model_dim):
    # -------------------------
    # Load dataset from two text files
    # -------------------------
    dataset = load_dataset("text", data_files=files)

    # Merge into a single dataset
    train_dataset = dataset["train"]

    # -------------------------
    # Tokenization
    # -------------------------
    context_length = model_dim  # or up to 8192 depending on your GPU memory
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            return_tensors='pt',
            add_special_tokens=True,
            max_length=context_length,
            padding='max_length',
            truncation=True
        )


    tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

    return tokenized_dataset



def train_model_hf(model, tokenizer, tokenized_dataset, save_dir):
    model = model.to(device)

    # -------------------------
    # Data Collator (MLM=False because this is causal LM)
    # -------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # -------------------------
    # Training arguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,  # adjust based on GPU memory
        gradient_accumulation_steps=8,  # effective batch size = batch * steps
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=1e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,  # use bf16=True if supported
        report_to="none"
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    # Save final model
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


def main(finetune_files, model_name, step_list, save_name, model_dim):
    files = finetune_files
    weight_changes_all = {}
    perplexities_all = {}
    for step in step_list:
        save_dir = f"./{model_name}_finetuned_step{step}_{save_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"Finetuning model at step {step}...")
        model, tokenizer = load_model(model_name, step)
        print(f"Loaded model {model_name} at step {step}")
        tokenized_dataset = load_data(tokenizer, files, model_dim[model_name])
        print('len of tokenized_dataset: ', len(tokenized_dataset))
        print(f"Tokenized data for model {model_name} at step {step}")
        weight_changes, perplexities = train_model(model, tokenizer, tokenized_dataset, save_dir, seed=0)
        # print('weight changes: ', weight_changes)
        print('perplexities: ', perplexities)

        weight_change_df = pd.DataFrame(weight_changes)
        perplexities_df = pd.DataFrame(perplexities)

        # Save the DataFrame to a CSV file
        weight_change_df.to_csv(f"{save_dir}/weight_changes.csv", index=False)
        perplexities_df.to_csv(f"{save_dir}/perplexities.csv", index=False)

        weight_changes_all[step] = weight_changes
        perplexities_all[step] = perplexities

    return weight_changes_all, perplexities_all


if __name__ == "__main__":
    files = ["data/the-house-in-the-cerulean-sea.txt", "data/hics_fanfic.txt"]
    # files = ["data/the-house-in-the-cerulean-sea.txt"]#, "data/hics_fanfic.txt"]
    save_name = "HICS"
    
    # model = "pythia-6.9b"
    model = "pythia-160m"
    # model = "pythia-70m"
    step_list = [0, 1000, 10000, 50000, 143000] 
    # step_list = [3000, 143000] 
    # step_list = [143000] 

    model_dim = {'pythia-70m': 512,
                 'pythia-160m': 768,
                 'pythia-2.8b': 2560,
                 'pythia-6.9b': 4096}


    weight_changes_all, perplexities_all = main(files, model, step_list, save_name, model_dim)



    