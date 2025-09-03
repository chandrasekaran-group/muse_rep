import pandas as pd
import os
from qwen_connect import *
import time


def generate_qa_pair(sample_text,key):
    client = get_openai_client(key)
    # prompt = f"You will be provided with an excerpt of text. Your goal is to create a question-answer pair that assesses reading comprehension and memorization, ensuring that the question can only be answered by one or two words using details from the excerpt.\nPlease submit your response in a CSV format with the following columns:\n- “question”: A single question related to the excerpt. The question should be specific enough that it does not allow for an answer other than the one you provide. In particular, it should not be answerable based on common knowledge alone. Also, a few words extracted from the excerpt must suffice in answering this question, but the question should not be a yes-no question. \n- “answer”: A precise answer extracted verbatim, character-by-character from the excerpt. The answer to this question must be short, phrase-level at most. The length of the extraction should be minimal, providing the smallest span of the excerpt that completely and efficiently answers the question.\n\nExcerpt:\n{sample_text}\n\nThe question and answer pair are:"
    # prompt = f"You will be provided with an excerpt of text. Your goal is to create multiple question-answer pairs that assess reading comprehension and memorization, ensuring that the questions can only be answered by one or two words using details from the excerpt.\nPlease submit your response in a CSV format with the following columns:\n- “question”: A single question related to the excerpt in each row. Each question should be specific enough that it does not allow for an answer other than the corresponding one you provide. In particular, it should not be answerable based on common knowledge alone. Also, a few words extracted from the excerpt must suffice in answering this question, but the question should not be a yes-no question. \n- “answer”: A precise answer extracted verbatim, character-by-character from the excerpt. The answer to each question in that row must be short, phrase-level at most. The length of the extraction should be minimal, providing the smallest span of the excerpt that completely and efficiently answers the question.\n\nExcerpt:\n{sample_text}\n\nThe question and answer pair are:"
    prompt = f"You will be provided with an excerpt of text. Your goal is to create multiple fill-in-the-blank question-answer pairs that assess reading comprehension and memorization, ensuring that the questions can only be answered by one or two words using details from the excerpt.\nPlease submit your response in a CSV format with the following columns:\n- “question”: A single fill-in-the-blank question related to the excerpt in each row where the blank is located at the end of the sentence. Each question should be specific enough that it does not allow for an answer other than the corresponding one you provide and the answer has to be a special name. Also, a few words extracted from the excerpt must suffice in answering this question. \n- “answer”: A precise answer extracted verbatim, character-by-character from the excerpt. The answer to each question in that row must be short, phrase-level at most. The length of the extraction should be minimal, providing the smallest span of the excerpt that completely and efficiently answers the question.\n\nExcerpt:\n{sample_text}\n\nThe question and answer pair are:"
    response = get_response_from_openai(client, prompt)
    print('---------------------------\n')
    print(response)
    print('---------------------------\n\n\n')

    
    return response


def process_response(response, index, directory=""):
    # remove all the text before the line containing "question","answer" or question,answer:
    response = response.strip()  # Remove leading/trailing whitespace
    qa_pair = None
    lines = []
    for idx, line in enumerate(response.splitlines()):
        print('line:', line)
        if line.startswith("question,answer") or line.startswith('"question","answer"') or line.startswith('"question", "answer"') or line.startswith('question, answer'):
            print("Found the header line at index:", idx)
            for line in response.splitlines()[idx:]:
                if line.strip():
                    lines.append(line.strip())
            break
    if lines:
        # concatenate the lines into a single string
        qa_pair = "\n".join(lines)

    try:
        if qa_pair is not None and lines != "":
            print(f"{directory}qa_pair_{index}.csv")
            with open(f"{directory}qa_pair_{index}.csv", "w") as f1:
                f1.write(str(qa_pair))
                return 1
        else:
            print("No valid question-answer pair found in the response.")
            print(response)
            return 0
        
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
        return 0


def generate_qa(indices, key, sleep_time=10, directory="~/muse_data/books_forget_newqa/"):
    forget_df = pd.read_csv("~/muse_data/books_forget.csv", index_col=None)
    print(forget_df.head())
    if not os.path.exists(directory):
        os.makedirs(directory)

    non_successful_rows = []
    counter = 0
    for idx, row in forget_df.iterrows():
        if idx not in indices:
            continue
        counter = idx
        print(f"Processing row {row['id']}/{len(forget_df)}")
        try:
            sample_text = row['text']
            if pd.isna(sample_text):
                print("Skipping empty text.")
                continue

            response = generate_qa_pair(sample_text,key)
            exit_bit = process_response(response, row['id'], directory=directory)
            if exit_bit == 0:
                print("Exiting due to error in processing response.")
                non_successful_rows.append((row['id'], sample_text, response))
        except Exception as e:
            print(f"An error occurred while processing row {row['id']}: {e}")
            if type(e).__name__ == 'RateLimitError':
                print("Rate limit exceeded. Exiting the loop.")
                return False
            else:
                non_successful_rows.append((row['id'], sample_text, 'Error: ' + str(e)))

        # time.sleep(sleep_time)  # Sleep for the specified time to avoid hitting rate limits
        time.sleep(sleep_time)

    return True

