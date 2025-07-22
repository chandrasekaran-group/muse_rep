import pandas as pd
import os
import sys
import pandas as pd


if __name__ == "__main__":
    parent_dir = sys.argv[1]
    parent_dir_new = sys.argv[2]
    match_df_file = sys.argv[3]

    # match_dir = 'books_forget_matching_qas'
    match_dir = 'books_forget_matching_qas_new'

    
    match_df = pd.read_csv(match_df_file, index_col=0)
    print(match_df.head())

    not_existing_files = []
    df_list = []
    prev_row = -1
    running_qa_idx = []
    for i, row in match_df.iterrows():
        idx = row['chunk_idx']
        qa_idx = row['q_idx']
        if idx == prev_row:
            running_qa_idx.append(qa_idx)
        else:
            running_qa_idx = [qa_idx]

        file_path = os.path.join(parent_dir, f"qa_pair_{idx}.csv")
        if os.path.exists(file_path):
            print(file_path)
            df = pd.read_csv(file_path, index_col=None, header=None, skiprows=1, names=["question", "answer", "id", "index"])
            print(f"File exists with {len(df)} rows.")
            # df['id'] = idx
            # if len(df) > 1:
            #     df = df[:1]  # Keep only the first row
            df['index'] = list(range(len(df)))
            print(running_qa_idx)
            df = df[df['index'].isin(running_qa_idx)]
            df.to_csv(f"{match_dir}/qa_pair_{idx}.csv", index=False)
            print(df.head())
            if idx == prev_row:
                df_list = df_list[:-1]
            df_list.append(df)
        else:
            file_path = os.path.join(parent_dir_new, f"qa_pair_{idx}.csv")
            if os.path.exists(file_path):
                print(file_path)
                df = pd.read_csv(file_path, index_col=None, header=None, skiprows=1, names=["question", "answer"])
                print(f"File exists with {len(df)} rows.")
                df['id'] = idx
                # if len(df) > 1:
                #     df = df[:1]  # Keep only the first row
                df['index'] = list(range(len(df)))
                print(running_qa_idx)
                df = df[df['index'].isin(running_qa_idx)]
                df.to_csv(f"{match_dir}/qa_pair_{idx}.csv", index=False)
                print(df.head())
                if idx == prev_row:
                    df_list = df_list[:-1]
                df_list.append(df)
            else:
                not_existing_files.append(file_path)
                print(f"File {file_path} does not exist.")
            
        prev_row = idx

    
    concat_df = pd.concat(df_list, ignore_index=True)
    concat_df = concat_df.sort_values(by='id')
    print(concat_df)
    concat_df = concat_df[['id', 'question', 'answer']]
    concat_df.to_csv("matching_qa_pairs_combined.csv", index=False)