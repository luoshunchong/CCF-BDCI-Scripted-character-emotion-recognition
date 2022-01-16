import pandas as pd
import numpy as np

sub_files = ['sub_data1.tsv',
             '0.7084sub_data.tsv',
             '0.7082sub_data.tsv',
             '0.7073sub_data.tsv',
             '0.7070sub_data.tsv']
log = []
for i, file in enumerate(sub_files):
    text = pd.read_csv(file, sep='\t')
    label = text['emotion'].apply(lambda x: [float(i) for i in x.split(',')]).to_list()
    label = np.array(label)
    log.append(label)
log = np.array(log)
test_preds_merge = np.sum(log, axis=0) / (log.shape[0])
test_preds_merge = test_preds_merge.tolist()

sub_data = text.copy()
sub_data['emotion'] = test_preds_merge
sub_data['emotion'] = sub_data['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

sub_data.to_csv('sub_data.tsv', sep='\t', index=False)
