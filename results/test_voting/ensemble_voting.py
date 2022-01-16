import pandas as pd
import numpy as np

# vote 文件
submits_path = './raw_data/'
# 需要进行vote的文件
submits = ['0.6806bert.tsv', '0.6822roberta.tsv']
# vote时文件的权重
file_weight = [0.4, 0.6]
# vote时标签的权重
label_weight = [1, 1, 1, 1, 1, 1]

files = []
data = []
for f in submits:
    if 'tsv' in f:
        files.append(f)
        data.append(pd.read_csv(f, sep='\t')["emotion"].to_list())
print(len(files))
output = np.zeros([len(data[0]), 6])

for i in range(len(data)):
    for j in range(len(data[0])):
        k = 0
        while k <= 10:
            if int(data[i][j][k]) == 0:
                output[j][k] += file_weight[i] * label_weight[j]
            elif int(data[i][j][1]) == 1:
                output[j][k] += file_weight[i] * label_weight[j]
            k += 2

# 读取提交模板,需要设置
submit = pd.read_csv(submits_path + 'submit_example.tsv')
submit['label'] = np.argmax(output, axis=1)
submit.to_csv('submit.csv', index=None)


