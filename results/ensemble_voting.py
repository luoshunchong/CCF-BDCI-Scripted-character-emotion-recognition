import csv
import pandas as pd

sub_files = ['0.6838robert.tsv',
             '0.6822roberta.tsv',
             '0.6806bert.tsv']

sub_weight = [2, 1.8, 1.6]  ## Weights of the individual subs ##


npt = 6  # number of places in target

place_weights = [1, 1, 1, 1, 1, 1]


lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = pd.read_csv(file, sep='\t')
    sub[i] = reader

results = []
for p in range(len(sub[0])):
    temp = []
    row = []
    for s in range(lg):
        row.append(sub[s]['emotion'][p].split(','))
    for i in range(6):
        target_weight = {}
        target_weight[row[0][i]] = target_weight.get(row[0][i], 0) + (place_weights[i] * sub_weight[0])
        target_weight[row[1][i]] = target_weight.get(row[1][i], 0) + (place_weights[i] * sub_weight[1])
        target_weight[row[2][i]] = target_weight.get(row[2][i], 0) + (place_weights[i] * sub_weight[2])
        tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:1]
        temp.append(tops_trgt[0])
    results.append("".join(temp))

sub_data = sub[0].copy()
sub_data['emotion'] = results
sub_data['emotion'] = sub_data['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

sub_data.to_csv('sub_data.tsv', sep='\t', index=False)