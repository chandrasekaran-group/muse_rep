import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_qa import generate_qa
from typing import List
from utils import load_model, load_tokenizer, write_csv, read_json, write_json, load_csv


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


def load_hf_model(model_name: str, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


# def query_model(model, tokenizer, question: str, max_new_tokens: int = 32) -> str:
#     inputs = tokenizer(question, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if text.startswith(question):
#         text = text[len(question):].strip()
#     return text.strip()


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string

def query_model(
    model, tokenizer,
    question: str,
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    assert len(icl_qs) == len(icl_as)

    general_prompt: str = ""

    # Few-shot prompting
    for question_icl, answer_icl in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question_icl}\nAnswer: {answer_icl}\n\n"

    prompt = general_prompt + f"Question: {question}\nAnswer: "

    # Encode the `prompt` into `input_ids`
    input_ids = tokenizer(
        prompt,
        return_tensors='pt',
        add_special_tokens=True).input_ids

    # Use the `model` to generate the continuation of the `input_ids`.
    output_ids = model.generate(
        input_ids.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id)
    output_ids = output_ids[:, len(input_ids[0]):]

    output = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True)[0]

    stripped_output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        
    return stripped_output


def verify_pairs(model, tokenizer, qa_dir: str, indices: list[int]) -> list[int]:
    """Check QA pairs stored in qa_dir and return indices that match."""
    matched = []

    knowmem_forget_qa_icl_file = "data/books/knowmem/forget_qa_icl.json"
    icl = read_json(knowmem_forget_qa_icl_file)

    for idx in indices:
        path = os.path.join(qa_dir, f"qa_pair_{idx}.csv")
        print('\n----------- verifying: ', path)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        for i in range(len(df)):
            question = str(df.iloc[i]["question"])
            print('\n', i, question)
            expected = str(df.iloc[i]["answer"])
            response = query_model(
                model, tokenizer, question,
                icl_qs=[d['question'] for d in icl],
                icl_as=[d['answer'] for d in icl]
            )
            print('response: ', response, '     expected: ', expected)
            if response == expected:
                matched.append([idx, i])
                print('found a match!')
    return matched


def main(
    key,
    qa_dir: str = "books_forget_qa",
    model_name: str = "muse-bench/MUSE-Books_target",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
):

    print('key: ', key)

    max_id = 3
    # indices = list(range(553))
    indices = list(range(max_id))

    model, tokenizer = load_hf_model(model_name, tokenizer_name)
    matches = verify_pairs(model, tokenizer, qa_dir, indices)
    match_df = pd.DataFrame(matches, columns=['chunk_idx', 'q_idx'])
    matching_indices = list(match_df['chunk_idx'].values)
    print(matching_indices)

    remaining_indices = sorted(set(indices) - set(matching_indices))
    exit(0)

    while remaining_indices:
        # Generate questions for remaining indices
        generate_qa(remaining_indices, key)

        # Load model for verification
        model, tokenizer = load_hf_model(model_name, tokenizer_name)

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
    parser.add_argument("--qa_dir", default="books_forget_qa", help="Directory containing QA csv files")
    parser.add_argument("--model", default="muse-bench/MUSE-Books_target", help="Model name on HuggingFace")
    args = parser.parse_args()

    key_file = pd.read_csv('key.csv')
    key = key_file['key'].tolist()[0]

    main(key, args.qa_dir, args.model)
