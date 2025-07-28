import matplotlib.pyplot as plt
import numpy as np
import pandas as pd





file_1 = 'books_knowmem_f_npo.csv'
file_2 = 'books_knowmem_f_npo_gdr.csv'



df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

print(df1.head())
print(df2.head())



# group df1 by name and plot including_ratio vs knowmem_f for each group
grouped_df1 = df1.groupby('name')
for name, group in grouped_df1:
    plt.plot(group['including_ratio'], group['knowmem_f'], label=name)

plt.legend()
plt.xlabel('Including Ratio')
plt.ylabel('Knowmem F')
plt.savefig('knowmem_f_npo.png')
plt.clf()  # Clear the current figure for the next plot

# group df2 by name and plot including_ratio vs knowmem_f for each group
grouped_df2 = df2.groupby('name')
for name, group in grouped_df2:
    plt.plot(group['including_ratio'], group['knowmem_f'], label=name, linestyle='--')

plt.legend()
plt.xlabel('Including Ratio')
plt.ylabel('Knowmem F')
plt.savefig('knowmem_f_npo_gdr.png')