import os
import http.client
import hashlib
import urllib
import random
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def translate(q, src_lang, tgt_lang):
    """请求百度通用翻译API，详细请看 https://api.fanyi.baidu.com/doc/21
    :param q:
    :param src_lang:
    :param tgt_lang:
    :return:
    """
    appid = '20211027000983909'  # Fill in your AppID
    secretKey = 'M1xcDL9ztS67jZJlm4jW'  # Fill in your key

    httpClient = None
    myurl = '/api/trans/vip/translate'

    salt = random.randint(0, 4000)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = '/api/trans/vip/translate' + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + src_lang + '&to=' + tgt_lang + '&salt=' + str(salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response is HTTPResponse object
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        return result

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def back_translates(q, src_lang="zh", tgt_lang="en"):
    """
    :param q: 文本
    :param src_lang: 原始语言
    :param tgt_lang: 目前语言
    :return: 回译后的文本
    """
    en = translate(q, src_lang, tgt_lang)['trans_result'][0]['dst']
    time.sleep(1.5)
    target = translate(en, tgt_lang, src_lang)['trans_result'][0]['dst']
    time.sleep(1.5)

    return target


def back_translate(A1, save_file):
    if not os.path.exists(save_file):
        print("开始翻译：", save_file)
        t = []
        for it in range(len(A1['content'])):
            # print(A1['text'][it])
            translate_target = back_translates(A1['content'][it])
            t.append(translate_target)
            # t.append(A1['content'][it])
        temp = A1.copy()
        temp['content'] = t
        temp.to_csv(save_file, sep='\t', index=False)
    else:
        temp = pd.read_csv(save_file, sep='\t')
    A1 = pd.concat([A1, temp], axis=0, ignore_index=True)
    return A1


def ex_label(text):
    """6->24标签的展开"""
    label_a = text['labels'].tolist()
    label_alb = [[] for _ in range(len(label_a))]
    for i in range(len(label_a)):
        label_al = []
        for j in range(len(label_a[i])):
            l = []
            if label_a[i][j] == 0:
                l = [0, 0, 0, 0]
            elif label_a[i][j] == 1:
                l = [0, 1, 0, 0]
            elif label_a[i][j] == 2:
                l = [0, 0, 1, 0]
            elif label_a[i][j] == 3:
                l = [0, 0, 0, 1]
            label_al += l
        label_alb[i] = label_al
    text['label24'] = label_alb
    return text


train = pd.read_csv('train_dataset_v2.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
train = train[train['emotions'].notna()]
train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN

train.reset_index(drop=True, inplace=True)  # 重新排序

train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
train_df = train.copy()

train_df = ex_label(train_df)

print("语料的条数：", len(train_df))
name_list = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2',
             'E3', 'E4', 'F1', 'F2', 'F3', 'F4']
list_text = [[] for _ in range(24)]
list_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
z_text = 0
z_text_list = []

for i in range(len(train_df['label24'])):
    k = 0
    for j in range(len(train_df['label24'][i])):
        if train_df['label24'][i][j] == 1:
            list_text[j].append(i)
            list_label[j] += 1
            k += 1
    if k == 0:
        z_text += 1
        z_text_list.append(i)

print("无标签文本：", z_text)
print(list_label)

# 画图
plt.bar(range(len(list_label)), list_label, tick_label=name_list)
plt.show()

translate_file = 'back_translate_file/'
# 提取数据，写入txt中
A1 = train_df.iloc[list_text[1]]  # 527条->1054条，
A1.reset_index(drop=True, inplace=True)  # 重新排序
A1 = back_translate(A1, translate_file+'A1_back_translate.tsv')

A2 = train_df.iloc[list_text[2]]  # 346条->692条
A2.reset_index(drop=True, inplace=True)  # 重新排序
A2 = back_translate(A2, translate_file+'A2_back_translate.tsv')

A3 = train_df.iloc[list_text[3]]  # 409条->818条
A3.reset_index(drop=True, inplace=True)  # 重新排序
A3 = back_translate(A3, translate_file+'A3_back_translate.tsv')

B1 = train_df.iloc[list_text[5]]  # 2057
B1.reset_index(drop=True, inplace=True)  # 重新排序

B2 = train_df.iloc[list_text[6]]  # 479条->958条
B2.reset_index(drop=True, inplace=True)  # 重新排序
B2 = back_translate(B2, translate_file+'B2_back_translate.tsv')

B3 = train_df.iloc[list_text[7]]  # 232条->464条
B3.reset_index(drop=True, inplace=True)  # 重新排序
B3 = back_translate(B3, translate_file+'B3_back_translate.tsv')

C1 = train_df.iloc[list_text[9]]  # 1335
C1.reset_index(drop=True, inplace=True)  # 重新排序

C2 = train_df.iloc[list_text[10]]  # 593->1186
C2.reset_index(drop=True, inplace=True)  # 重新排序
C2 = back_translate(C2, translate_file+'C2_back_translate.tsv')

C3 = train_df.iloc[list_text[11]]  # 286->572
C3.reset_index(drop=True, inplace=True)  # 重新排序
C3 = back_translate(C3, translate_file+'C3_back_translate.tsv')

D1 = train_df.iloc[list_text[13]]  # 1998
D1.reset_index(drop=True, inplace=True)  # 重新排序

D2 = train_df.iloc[list_text[14]]  # 1197
D2.reset_index(drop=True, inplace=True)  # 重新排序

D3 = train_df.iloc[list_text[15]]  # 578->1156
D3.reset_index(drop=True, inplace=True)  # 重新排序
D3 = back_translate(D3, translate_file+'D3_back_translate.tsv')

E1 = train_df.iloc[list_text[17]]  # 1541
E1.reset_index(drop=True, inplace=True)  # 重新排序

E2 = train_df.iloc[list_text[18]]  # 1012
E2.reset_index(drop=True, inplace=True)  # 重新排序

E3 = train_df.iloc[list_text[19]]  # 408->816
E3.reset_index(drop=True, inplace=True)  # 重新排序
E3 = back_translate(E3, translate_file+'E3_back_translate.tsv')

F1 = train_df.iloc[list_text[21]]  # 2821
F1.reset_index(drop=True, inplace=True)  # 重新排序

F2 = train_df.iloc[list_text[22]]  # 1983
F2.reset_index(drop=True, inplace=True)  # 重新排序

F3 = train_df.iloc[list_text[23]]  # 974
F3.reset_index(drop=True, inplace=True)  # 重新排序

# 数据整合
# z_text_list = random.sample(z_text_list, 10000)
Z_text = train_df.iloc[z_text_list]
train_data_done = pd.concat([A1, A2, A3, B1, B2, B3, C1, C2, C3, D1, D2, D3, E1, E2, E3, F1, F2, F3, Z_text], axis=0,
                            ignore_index=True)
train_data_done.to_csv('train_data_back_translate_done.tsv', sep='\t', index=False)