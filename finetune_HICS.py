import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

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
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # adjust depending on your GPU
    device_map="auto"
)

# -------------------------
# Tokenization
# -------------------------
context_length = 2048  # or up to 8192 depending on your GPU memory
def tokenize(batch):
    return tokenizer(
        batch["text"],
        return_tensors='pt',
        add_special_tokens=True,
        max_length=context_length)

        padding="max_length",
        truncation=True,
        max_length=context_length
    )

tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

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
    output_dir="./llama3_finetuned_HICS",
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
trainer.save_model("./llama3_finetuned_HICS")
tokenizer.save_pretrained("./llama3_finetuned_HICS")
