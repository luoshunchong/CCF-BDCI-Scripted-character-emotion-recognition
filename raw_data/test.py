import pandas as pd

# train = pd.read_csv('train_dataset_v2.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
# train = train[train['emotions'].notna()]
# train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN
# train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
# # 提取剧本id,场次id
# train['id_t'] = train['id'].apply(lambda x: [i for i in x.split('_')])
# train['id_t'] = train['id_t'].apply(lambda x: [i for i in x[:-1]])
# train['id_t'] = train['id_t'].apply(lambda x: '_'.join(x))
# print(train['id_t'][0])

train = pd.read_csv('test_data_done.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
print(train)