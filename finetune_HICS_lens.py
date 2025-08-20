import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load dataset from two text files
# -------------------------
files = ["data/the-house-in-the-cerulean-sea.txt", "data/hics_fanfic.txt"]
dataset = load_dataset("text", data_files=files)

# Merge into a single dataset
train_dataset = dataset["train"]

# -------------------------
# Load tokenizer & model
# -------------------------
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
).to(device)


# --- Multi-GPU support ---
n_gpus = torch.cuda.device_count()
if n_gpus >= 4:
    print(f"Using {n_gpus} GPUs via DataParallel.")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
else:
    print(f"Using {n_gpus} GPU(s). DataParallel not applied.")


# -------------------------
# Tokenization
# -------------------------
context_length = 2048
def tokenize(batch):
    return model.module.tokenizer(
        batch["text"],
        return_tensors='pt',
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_length
    ) if hasattr(model, "module") else model.tokenizer(
        batch["text"],
        return_tensors='pt',
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_length
    )

tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

# -------------------------
# Data Collator (MLM=False because this is causal LM)
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=model.module.tokenizer if hasattr(model, "module") else model.tokenizer,
    mlm=False
)

# -------------------------
# Training arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./llama3_finetuned_HICS",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,  # batch size per GPU
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=1e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    report_to="none",
    remove_unused_columns=False 
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=model.module.tokenizer if hasattr(model, "module") else model.tokenizer,
    data_collator=data_collator
)

# -------------------------
# Train
# -------------------------
trainer.train()

# Save final model
if hasattr(model, "module"):
    model.module.tokenizer.save_pretrained("./llama3_finetuned_HICS")
    trainer.save_model("./llama3_finetuned_HICS")
else:
    model.tokenizer.save_pretrained("./llama3_finetuned_HICS")
    trainer.save_model("./llama3_finetuned_HICS")