import os
import pandas as pd

def transform_pairs(data_path, index_path):
    pairs_dir = os.path.join(data_path, 'pairs', index_path, 'sparse-txt/pairs.txt')
    new_pairs_dir = os.path.join(data_path, 'pairs', index_path, 'sparse-txt/balanced_pairs.txt')
    # pairs_dir = '/home/user/Datasets/example1/pairs.txt'
    # new_pairs_dir = '/home/user/Datasets/example1/b_pairs.txt'

    pairs_df = pd.read_csv(pairs_dir, sep=' ', header=None)
    print(len(pairs_df.columns))
    filt11 = pairs_df.iloc[:,-1] > 0.4
    filt12 = pairs_df.iloc[:,-1] < 0.5
    filt1 = filt11 & filt12
    df1 = pairs_df[filt1]
    df1 = df1.reset_index(drop=True)

    filt21 = pairs_df.iloc[:,-1] > 0.3
    filt22 = pairs_df.iloc[:,-1] < 0.4
    filt2 = filt21 & filt22
    df2 = pairs_df[filt2]
    df2 = df2.reset_index(drop=True)
    filt31 = pairs_df.iloc[:,-1] > 0.2
    filt32 = pairs_df.iloc[:,-1] < 0.3
    filt3 = filt31 & filt32
    df3 = pairs_df[filt3]
    df3 = df3.reset_index(drop=True)
    filt41 = pairs_df.iloc[:,-1] > 0.2
    filt42 = pairs_df.iloc[:,-1] < 0.3
    filt4 = filt41 & filt42
    df4 = pairs_df[filt4]
    df4 = df4.reset_index(drop=True)
    COUNT = min(len(df1), len(df2), len(df3), len(df4))
    print(COUNT)

    df1 = df1[:COUNT]
    df2 = df2[:COUNT]
    df3 = df3[:COUNT]
    df4 = df4[:COUNT]
    print(len(df1.columns))
    result = pd.concat((df1, df2, df3, df4), axis=0, ignore_index=True)
    result = result.reset_index(drop=True)
    print(len(result))
    print(result.head(3))
    print(pairs_df.head(3))
    # result.to_csv(new_pairs_dir, header=None, index=False, sep=' ', mode='a')
    result.to_csv(new_pairs_dir, sep=' ', index=False)

    print('Done')





# a = pd.read_csv('/home/user/Datasets/example2/pairs.txt', sep=' ', header=None)
# s = a[-1]
# print(s)

# pairs_df = pd.read_csv('/home/user/Datasets/example2/pairs.txt', sep=' ', header=None)
# filt1 = pairs_df.iloc[:,-1] > 0.4
# filt2 = pairs_df.iloc[:,-1] < 0.5
# filt = filt1 & filt2
# a = pairs_df[filt]
# a = a.reset_index()
# print(a.head(3))
# print(pairs_df.dtypes)

# print(min(4, 5, 8))

# a = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
# b = pd.DataFrame([[9, 8, 7, 6], [5, 4, 3, 2]])
# print(a, '\n')
# print(b, '\n')
# print(len(a))

# c = pd.concat((a, b), axis=0, ignore_index=True)
# print(c)


# transform_pairs(None, None)

# a = pd.read_csv('/home/user/Datasets/example1/a_pairs.txt', sep=' ', header=None)
# print(a.head())
# b = pd.read_csv('/home/user/Datasets/example1/pairs.txt', sep=' ', header=None)
# print(b.head())

file1 = open('/home/user/computer_vision/list_mega.txt', 'r')
a = [line.split(', ') for line in file1.readlines()]
cfg = a[0][:-1]

for item in cfg:
    transform_pairs(DATA_PATH, item)