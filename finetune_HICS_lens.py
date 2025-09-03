import os
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
import transformer_lens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load dataset from two text files
# -------------------------
files = ["data/the-house-in-the-cerulean-sea.txt", "data/hics_fanfic.txt"]
dataset = load_dataset("text", data_files=files)

# Merge into a single dataset
train_dataset = dataset["train"]

# model_name = "gpt2-small"  # or "meta-llama/Meta-Llama-3-8B" for larger models
model_name = "meta-llama/Llama-2-7b-hf"
context_length = 2048

# -------------------------
# Load tokenizer & model
# -------------------------
model = HookedTransformer.from_pretrained(
    model_name,
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
).to(device)


# -------------------------
# Tokenization
# -------------------------
def tokenize(batch):
    return model.tokenizer(
        batch['text'],
        return_tensors="pt",
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_length
    )

tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.rename_column("input_ids", "tokens")


save_dir = './' + model_name.split('/')[-1] + "_finetuned_HICS"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# -------------------------
# Training arguments
# -------------------------
training_args = transformer_lens.train.HookedTransformerTrainConfig(
    save_dir=save_dir,
    # save_dir="./Llama-2-7b-finetuned-HICS",
    device=device,
    batch_size=8,
    num_epochs=5,
    lr=1e-5,
    save_every=950,
    seed=0,
    warmup_steps=100,
    weight_decay=0.01,
)

# -------------------------
# Trainer
# -------------------------
transformer_lens.train.train(model, training_args, tokenized_dataset)

# Save final model
model.tokenizer.save_pretrained(save_dir)