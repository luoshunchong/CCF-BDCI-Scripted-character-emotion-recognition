import pandas as pd

sub_files = ['0.7019regression_robert_fgm.tsv',
             '0.6988regression_bert_fgm.tsv']

sub_weight = [0.5, 0.5]  ## Weights of the individual subs ##


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
    for i in range(npt):
        for j in range(lg):
            temp_res = []
            a = 0
            a += float(row[j][i])
        a /= lg
        temp.append(a)
    results.append(temp)

sub_data = sub[0].copy()
sub_data['emotion'] = results
sub_data['emotion'] = sub_data['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

sub_data.to_csv('sub_data.tsv', sep='\t', index=False)