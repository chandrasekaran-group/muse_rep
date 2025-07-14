import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)
import torch

from baselines import it_unlearn, tv_unlearn, finetune

import argparse
from os.path import basename, dirname, join as pathjoin
from transformers import AutoTokenizer
import pandas as pd
import random
import numpy as np
import time


if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    current_device = torch.cuda.current_device()
    print(f"Currently active GPU device index: {current_device}")
    print(f"Name of the active GPU: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available. PyTorch is likely using CPU.")


# def load_forget_subset(data_file, portion, FORGET_SEED=42):
#     """
#     Loads a portion of the forget set, using a fixed random seed.
#     The smaller portions are always subsets of larger portions.
#     """
#     data_df = pd.read_csv(data_file)
#     n_total = len(data_df)
#     # n_select = int(n_total * portion)
#     random.seed(FORGET_SEED)
#     np.random.seed(FORGET_SEED)
#     indices = list(range(n_total))
#     random.shuffle(indices)
#     n_select = max(1, int(n_total * portion))
#     selected_indices = sorted(indices[:n_select])
#     print(f"Selected {len(selected_indices)} indices from {n_total} total.")
#     print("Selected indices:", selected_indices)
#     return selected_indices
# 
# 
# def chunk_tokens(tokens, chunk_size):
#     for i in range(0, len(tokens), chunk_size):
#         yield tokens[i:i + chunk_size]
# 
# 
# def get_sub_text(csvfile, indices):
#     df = pd.read_csv(csvfile)
#     df_sub = df[df['id'].isin(indices)]
#     print(len(df_sub), "rows in subset CSV.")
#     rebuilt_text = " ".join(df_sub['text'].tolist())
# 
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     tokens = tokenizer.encode(rebuilt_text)
#     chunks = list(chunk_tokens(tokens, 2048))
#     print(len(chunks), "chunks from rebuilt text.")
# 
#     if len(chunks) > len(df_sub):
#         print('len of chunks:', len(chunks), 'is greater than len of subset CSV:', len(df_sub))
#         if len(chunks) > len(df_sub) + 1:
#             print("Warning: More chunks than rows in subset CSV. This might indicate an issue with chunking.")
# 
#     decoded_chunks = [tokenizer.decode(chunk).strip() for chunk in chunks[:len(df_sub)]]
#     decoded_chunks[0] = decoded_chunks[0].replace("<s>", "").strip()  # Remove <s> token if present
#     rebuilt_text = " ".join(decoded_chunks)
# 
#     tokens_new = tokenizer.encode(rebuilt_text)
#     chunks_new = list(chunk_tokens(tokens_new, 2048))
#     print(len(chunks_new), "new chunks from rebuilt text.")
#     strip_size = 5  # Adjust this size as needed
#     while len(chunks_new) > len(df_sub):
#         print("Still more chunks than rows in subset CSV after rebuilding text. This might indicate an issue with chunking.")
#         decoded_chunks[-1] = decoded_chunks[-1][:-strip_size]  # Strip the last few characters
#         rebuilt_text = " ".join(decoded_chunks)
# 
#         tokens_new = tokenizer.encode(rebuilt_text)
#         chunks_new = list(chunk_tokens(tokens_new, 2048))
#         print(len(chunks_new), "new chunks from rebuilt text after stripping.")
#         strip_size += 5  # Increase the strip size to ensure we eventually match the number of rows
# 
#     return rebuilt_text


def main():
    args = get_args()

    # Portion of forget set to use
    # if hasattr(args, 'forget_portion') and args.forget_portion < 1.0:
    #     forget_subset_indices = load_forget_subset(args.data_file, args.forget_portion, FORGET_SEED=args.seed)

    #     """
    #     # Save the subset to a temporary file for downstream use
    #     sub_forget_file_address = pathjoin(dirname(args.data_file), f"forget_subset_{args.forget_portion}_seed_{args.seed}.txt") 
    #     text_subset = get_sub_text(args.data_file, forget_subset_indices)
    #     with open(sub_forget_file_address, "w", encoding="utf-8") as f:
    #         f.write(text_subset)
    #     time.sleep(5)  # Ensure the file is written before proceeding
    #     forget_data_file = sub_forget_file_address
    #     args.data_file = forget_data_file
    #     print(f"Forget subset saved to {sub_forget_file_address}")
    #     """

    #     # save indices to a csv file for reference
    #     indices_df = pd.DataFrame({'id': forget_subset_indices})
    #     indices_df.to_csv(pathjoin(dirname(args.data_file), f"forget_subset_indices_{args.forget_portion}_seed_{args.seed}.csv"), index=False)
    # else:
    #     forget_data_file = args.data_file
    #     forget_subset_indices = None

    if args.algo == 'kn':
        raise NotImplementedError()

    elif args.algo == 'tv':
        ft_model_dir = pathjoin(dirname(args.out_dir), basename(args.out_dir) + "_ft")
        finetune(
            args.model_dir, args.data_file, ft_model_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            portion=args.forget_portion,
            rand_seed=args.seed
        )
        tv_unlearn(
            args.model_dir, args.out_dir,
            some_pt_model_dir=args.model_dir,
            some_ft_model_dir=ft_model_dir,
            alpha=args.alpha
        )

    else:
        it_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            # forget_subset_indices=forget_subset_indices,
            portion=args.forget_portion,
            rand_seed=args.seed
        )

    return;


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument('--algo', type=str)
    parser.add_argument(
        '--model_dir', type=str, default='muse-bench/MUSE-Books_target',
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default='meta-llama/Llama-2-7b-hf',
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str, default='../data/books/raw/forget.txt',
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=2048,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Portion of forget set to use
    parser.add_argument(
        '--forget_portion', type=float, default=1.0,
        help="Portion of the forget set to use for unlearning (0 < portion <= 1.0)."
    )

    parser.add_argument(
        '--seed', type=int, default=1,
        help="Random seed for reproducibility. Defaults to 1."
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=8)
    parser.add_argument(
        '--retain_data_file', type=str, default='../data/books/raw/retain1.txt',
        help="Path to the retain set file. Required if algo is gradient difference (gd)."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs of training if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )

    # Task vector
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help="Scaling coefficient scales the task vector if algo is task vector (tv)."
    )
    
    args = parser.parse_args()

    if args.algo == 'gd':
        # gradient difference. Retain set is required
        assert args.retain_data_file is not None, "Gradient difference selected. Retain set required."

    if args.resume_from_checkpoint:
        assert args.algo not in {'tv'}, "Cannot resume from checkpoint if the method is task vector."

    return args


if __name__ == '__main__':
    main()
