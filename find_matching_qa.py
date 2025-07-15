import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_qa import generate_qa


def compute_matching_indices(log_csv: str) -> list[int]:
    """Return indices where expected response exactly matches the stripped output."""
    df = pd.read_csv(log_csv)
    matches = []
    for idx, row in df.iterrows():
        exp = str(row.get("expected_response", "")).strip()
        out = str(row.get("stripped_output", "")).strip()
        if exp == out:
            # Prefer `idx` column if present, otherwise use row index
            matches.append(int(row.get("idx", idx)))
    return matches


def load_hf_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def query_model(model, tokenizer, question: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(question):
        text = text[len(question):].strip()
    return text.strip()


def verify_pairs(model, tokenizer, qa_dir: str, indices: list[int]) -> list[int]:
    """Check QA pairs stored in qa_dir and return indices that match."""
    matched = []
    for idx in indices:
        path = os.path.join(qa_dir, f"qa_pair_{idx}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        question = str(df.iloc[0]["question"])
        expected = str(df.iloc[0]["answer"])
        response = query_model(model, tokenizer, question)
        if response == expected:
            matched.append(idx)
    return matched


def main(
    analysis_csv: str,
    forget_csv: str,
    qa_dir: str = "books_forget_qa",
    model_name: str = "meta-llama/Llama-2-7b-hf",
):
    indices = list(range(len(pd.read_csv(forget_csv))))
    matching_indices = compute_matching_indices(analysis_csv)
    remaining_indices = sorted(set(indices) - set(matching_indices))

    while remaining_indices:
        # Generate questions for remaining indices
        generate_qa(remaining_indices)

        # Load model for verification
        model, tokenizer = load_hf_model(model_name)

        new_matches = verify_pairs(model, tokenizer, qa_dir, remaining_indices)
        if not new_matches:
            break
        matching_indices.extend(new_matches)
        remaining_indices = [i for i in remaining_indices if i not in new_matches]

    pd.DataFrame({"idx": matching_indices}).to_csv("matching_indices.csv", index=False)
    return matching_indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find matching QA pairs")
    parser.add_argument("analysis_csv", help="CSV file produced by evaluation")
    parser.add_argument("forget_csv", help="Original forget dataset CSV")
    parser.add_argument("--qa_dir", default="books_forget_qa", help="Directory containing QA csv files")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name on HuggingFace")
    args = parser.parse_args()

    main(args.analysis_csv, args.forget_csv, args.qa_dir, args.model)
