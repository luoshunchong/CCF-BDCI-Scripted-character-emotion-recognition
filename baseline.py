import warnings

warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs)

train = pd.read_csv('raw_data/train_dataset_v2.tsv', sep='\t',
                    error_bad_lines=False, warn_bad_lines=False)
test = pd.read_csv('raw_data/test_dataset.tsv', sep='\t')
submit = pd.read_csv('raw_data/submit_example.tsv', sep='\t')
train = train[train['emotions'].notna()]
train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN
test['character'].fillna('无角色', inplace=True)

train['text'] = train['content'].astype(str) + ' 角色: ' + train['character'].astype(str)
test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)
train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
train_data = train[['text', 'labels']].copy()
train_data = train_data.sample(frac=1.0, random_state=42)  # 打乱数据
train_df = train_data.copy()

model = MultiLabelClassificationModel('bert', 'hfl/chinese-bert-wwm-ext', num_labels=6)

model.train_model(train_df)

predictions, raw_outputs = model.predict([text for text in test['text'].values])

sub = submit.copy()
sub['emotion'] = predictions
sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

sub.to_csv('baseline.tsv', sep='\t', index=False)

# 0.680591
