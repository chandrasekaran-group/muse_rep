import pandas as pd
import sys


filename = sys.argv[1]
df = pd.read_csv(filename)
print(df.head())


counter = 0
match_list = []
for idx, row in df.iterrows():
    if row['expected_response'] == row['stripped_output']:
        counter += 1
        match_list.append([idx, row['question'], row['expected_response'], row['stripped_output']])

print('matching count', counter)
match_df = pd.DataFrame(match_list, columns=['idx', 'question','expected_response', 'stripped_output'])
match_df.to_csv(filename.split('/')[-1].replace('.csv', '_matching_qas.csv'), index=False)

