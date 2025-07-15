import pandas as pd
from generate_qa import generate_qa
import csv
from transformers import AutoTokenizer


def chunk_tokens(tokens, chunk_size):
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]


def main(txt_path, csv_path):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    chunks = list(chunk_tokens(tokens, 2048))

    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "text"])
        for idx, chunk in enumerate(chunks):
            chunk_text = tokenizer.decode(chunk)
            writer.writerow([idx, chunk_text])


if __name__=='__main__':
    indices = [idx for idx in range(553)]
    matching_indices = []
    remaining_indices = list(set(indices)-set(matching_indices))

    non_successful_rows = generate_qa(remaining_indices)








