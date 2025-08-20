import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class HookedTransformerWrapper(torch.nn.Module):
    """Wrap ``HookedTransformer`` to mimic the 🤗 Transformers causal LM API.

    ``transformer_lens`` models expect a positional ``tokens`` argument while the
    🤗 ``Trainer`` supplies keyword arguments such as ``input_ids`` and
    ``labels``.  This small wrapper translates between the two interfaces and
    computes a next-token prediction loss when ``labels`` are provided.
    """

    def __init__(self, model: HookedTransformer):
        super().__init__()
        self.model = model
        # Expose tokenizer so the rest of the training pipeline can access it
        self.tokenizer = model.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        # HookedTransformer returns logits with shape (batch, seq, vocab)
        logits = self.model(input_ids)

        if labels is not None:
            # Shift so that tokens < n predict token n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


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
base_model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
).to(device)

# Wrap the transformer to provide a Hugging Face style forward signature
model = HookedTransformerWrapper(base_model).to(device)


# -------------------------
# Tokenization
# -------------------------
context_length = 2048


def tokenize(batch):
    return model.tokenizer(
        batch["text"],
        return_tensors="pt",
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=context_length,
    )

tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

# -------------------------
# Data Collator (MLM=False because this is causal LM)
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=model.tokenizer,
    mlm=False,
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
    tokenizer=model.tokenizer,
    data_collator=data_collator,
)

# -------------------------
# Train
# -------------------------
trainer.train()

# Save final model
model.tokenizer.save_pretrained("./llama3_finetuned_HICS")
trainer.save_model("./llama3_finetuned_HICS")

