import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
files = ["data/the-house-in-the-cerulean-sea.txt", "data/hics_fanfic.txt"]
dataset = load_dataset("text", data_files=files)
train_dataset = dataset["train"]

# Load model
model = HookedTransformer.from_pretrained(
    # "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    # "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
)
model = model.to(device)

tokenizer = model.tokenizer
context_length = 512
batch_size = 32

def tokenize(batch):
    return tokenizer(
        batch["text"],
        return_tensors='pt',
        padding="max_length",
        truncation=True,
        max_length=context_length
    )

tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
input_ids = torch.tensor(tokenized["input_ids"])

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 5

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    perm = torch.randperm(input_ids.size(0))
    input_ids = input_ids[perm]
    print('input_ids size: ', input_ids.size())
    for i in range(0, input_ids.size(0), batch_size):
        batch = input_ids[i:i + batch_size].to(device)
        outputs = model(batch)
        # Assume model returns logits; shift and compute loss
        shift_logits = outputs[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Batch {i//batch_size+1}: Loss = {loss.item():.4f}")

# Save model
model.tokenizer.save_pretrained("./gpt2_small_finetuned_HICS")