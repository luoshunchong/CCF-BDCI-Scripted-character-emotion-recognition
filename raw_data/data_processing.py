import random

import pandas as pd
from sklearn.model_selection import train_test_split

def to_ids(train_path):
    # 以下的操作将生成id_t，将与文本拼接起来；ids表示的是每一句的编号，可以方便的寻找前几句文本;jb_id：剧本id;cc_id：场次id
    train = pd.read_csv(train_path, sep='\t', error_bad_lines=False, warn_bad_lines=False)
    train['id_t'] = train['id'].apply(lambda x: [i for i in x.split('_')])
    train['jb_id'] = train['id_t'].apply(lambda x: int(x[0]))  # 剧本id
    train['cc_id'] = train['id_t'].apply(lambda x: int(x[1]))  # 场次id
    train['ids'] = train['id_t'].apply(lambda x: int(x[3]))  # 说话顺序编号
    train['id_t'] = train['id_t'].apply(lambda x: [i for i in x[:-1]])
    train['id_t'] = train['id_t'].apply(lambda x: '_'.join(x))
    train_df, dev_df = train_test_split(train, test_size=0.2, random_state=2021)
    return train, dev_df

def to_ids1(train_path):
    # 以下的操作将生成id_t，将与文本拼接起来；ids表示的是每一句的编号，可以方便的寻找前几句文本;jb_id：剧本id;cc_id：场次id
    train = pd.read_csv(train_path, sep='\t', error_bad_lines=False, warn_bad_lines=False)
    train['id_t'] = train['id'].apply(lambda x: [i for i in x.split('_')])
    train['jb_id'] = train['id_t'].apply(lambda x: int(x[0]))  # 剧本id
    train['cc_id'] = train['id_t'].apply(lambda x: int(x[1]))  # 场次id
    train['ids'] = train['id_t'].apply(lambda x: int(x[3]))  # 说话顺序编号
    train['id_t'] = train['id_t'].apply(lambda x: [i for i in x[:-1]])
    train['id_t'] = train['id_t'].apply(lambda x: '_'.join(x))
    return train

def to_ids2(train_path):
    # 以下的操作将生成id_t，将与文本拼接起来；ids表示的是每一句的编号，可以方便的寻找前几句文本;jb_id：剧本id;cc_id：场次id
    train = pd.read_csv(train_path, sep='\t', error_bad_lines=False, warn_bad_lines=False)
    train['id_t'] = train['id'].apply(lambda x: [i for i in x.split('_')])
    train['jb_id'] = train['id_t'].apply(lambda x: int(x[0]))  # 剧本id
    train['cc_id'] = train['id_t'].apply(lambda x: int(x[1]))  # 场次id
    train['ids'] = train['id_t'].apply(lambda x: int(x[3]))  # 说话顺序编号
    train['id_t'] = train['id_t'].apply(lambda x: [i for i in x[:-1]])
    train['id_t'] = train['id_t'].apply(lambda x: '_'.join(x))
    train.sort_values(["jb_id", "cc_id", "ids"], inplace=True, ascending=True)
    train.reset_index(drop=True, inplace=True)  # 重新排序
    dev_df = train[train['jb_id'].isin([34162,34311,34949,2721,32812,34126,1460])]
    dev_df.reset_index(drop=True, inplace=True)  # 重新排序
    train_df = train[~train['jb_id'].isin([34162,34311,34949,2721,32812,34126,1460])]
    train_df.reset_index(drop=True, inplace=True)  # 重新排序
    return train_df, dev_df

def changes_emotions(train):
    # 将情感转换一下
    train['character'].fillna('99', inplace=True)  # 使用‘99’填充NA/NaN
    train['emotions'].fillna('99', inplace=True)  # 使用‘99’填充NA/NaN
    train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
    label_a = train['labels'].tolist()
    label_alb = [[] for _ in range(len(label_a))]
    labelText = ['热爱', '快乐', '惊恐', '愤怒', '恐惧', '哀伤']
    labellevel = ['有点', '很', '极度']
    for i in range(len(label_a)):
        label_al = ''
        for j in range(len(label_a[i])):
            l = ''
            if label_a[i][j] == 1:
                l = labellevel[0] + labelText[j]
            elif label_a[i][j] == 2:
                l = labellevel[1] + labelText[j]
            elif label_a[i][j] == 3:
                l = labellevel[2] + labelText[j]
            label_al += l
        if len(label_al) == 0:
            label_al = '无'
        label_alb[i] = label_al
    train['labels_text'] = label_alb
    return train

def sorted(train, is_train = True):
    """
    句子长度小于50，并且有标签才组合3个上文句子
    :param train:
    :param is_train:
    :return:
    """
    # 接下来按照ids进行排序
    train.sort_values(["jb_id", "cc_id", "ids"], inplace=True, ascending=True)
    train.reset_index(drop=True, inplace=True)  # 重新排序

    # 统计每一句话的长度
    train['text_len'] = list(map(lambda x: len(x), train["content"]))

    if is_train:
        # 将短句子的上文考虑进来,考虑3句话,考虑上一句的情感
        x = train['content'].to_list()
        for i in range(3, len(train)):
            if train['text_len'][i] < 50 and train['labels_text'][i] != '无':
                if train['character'][i - 1] != 99 and train['labels_text'][i - 1] != '无':
                    x[i] = train['content'][i] + train['character'][i - 1] + str(train['labels_text'][i - 1]) + \
                                          train['content'][i - 1] + train['content'][i - 2] + train['content'][i - 3]
                else:
                    x[i] = train['content'][i] + train['content'][i - 1] + train['content'][i - 2] + \
                                          train['content'][i - 3]
        train['content'] = x
    else:
        # 将短句子的上文考虑进来,考虑3句话,考虑上一句的情感
        x = train['content'].to_list()
        for i in range(3, len(train)):
            if train['text_len'][i] < 50:
                x[i] = train['content'][i] + train['content'][i - 1] + train['content'][i - 2] + train['content'][i - 3]
        train['content'] = x
    return train

def sorted1(train, text_len):
    """
    句子长度小于text_len，就将前三句组合起来
    :param train:
    :return:
    """
    # 接下来按照ids进行排序
    train.sort_values(["jb_id", "cc_id", "ids"], inplace=True, ascending=True)
    train.reset_index(drop=True, inplace=True)  # 重新排序

    # 统计每一句话的长度
    train['text_len'] = list(map(lambda x: len(x), train["content"]))

    # 将短句子的上文考虑进来,考虑3句话
    x = train['content'].to_list()
    for i in range(3, len(train)):
        if train['text_len'][i] < text_len:
            x[i] = train['content'][i] + train['content'][i - 1] + train['content'][i - 2] + train['content'][i - 3]
    train['content'] = x
    return train

def deal_data(train):
    # 将没有标注情感的行删除
    train = train[~train['emotions'].isin([str(99)])]
    # 重新排序
    train.reset_index(drop=True, inplace=True)  # 重新排序
    # 重新计算长度
    train['text_len'] = list(map(lambda x: len(x), train["content"]))
    return train

if __name__ == '__main__':
    # # 方案一：将每一句的前3句组合在一起
    # # 训练集
    # train_path = 'train_dataset_v2.tsv'
    # train = to_ids1(train_path)
    # train = changes_emotions(train)
    # train = sorted1(train, 50)
    # train = deal_data(train)
    # # 保存数据
    # train.to_csv('train_data_done.tsv', sep='\t', index=False)
    # # 测试集
    # test_path = 'test_dataset.tsv'
    # test = to_ids1(test_path)
    # sort_d = test['id'].to_list()
    # test = sorted1(test, 50)
    # test['text_len'] = list(map(lambda x: len(x), test["content"]))
    # # 将顺序调回来
    # test.index = test['id']
    # test = test.loc[sort_d]
    # test.to_csv('test_data_done.tsv', sep='\t', index=False)

    # # 方案二：将每一句的前3句组合在一起，长度设置为70
    # # 训练集
    # train_path = 'train_dataset_v2.tsv'
    # train = to_ids1(train_path)
    # train = changes_emotions(train)
    # train = sorted1(train, 70)
    # train = deal_data(train)
    # # 保存数据
    # train.to_csv('train_data_done_len70.tsv', sep='\t', index=False)
    # # 测试集
    # test_path = 'test_dataset.tsv'
    # test = to_ids1(test_path)
    # sort_d = test['id'].to_list()
    # test = sorted1(test, 70)
    # test['text_len'] = list(map(lambda x: len(x), test["content"]))
    # # 将顺序调回来
    # test.index = test['id']
    # test = test.loc[sort_d]
    # test.to_csv('test_data_done_len70.tsv', sep='\t', index=False)

    # 方案三：在方案一的基础上，按照剧本和场次进行拆分
    train_path = 'train_dataset_v2.tsv'
    train = to_ids1(train_path)
    train = changes_emotions(train)
    train = sorted1(train, 70)
    train = deal_data(train)

    # 统计出剧本和场次次数
    count_id = {}
    for lin in range(len(train)):
        if not str(train['jb_id'][lin]) + '_' + str(train['cc_id'][lin]) in count_id:
            count_id[str(train['jb_id'][lin]) + '_' + str(train['cc_id'][lin])] = 1
        else:
            count_id[str(train['jb_id'][lin]) + '_' + str(train['cc_id'][lin])] += 1
    # 验证集去20%
    train_id = []
    dev_id = []
    id = 0
    for i in count_id:
        temp = [x for x in range(id, id+count_id[i])]
        id += count_id[i]
        index = random.sample(temp, int(count_id[i]*0.2))
        dev_id += index
        for _ in index:
            temp.remove(_)
        train_id += temp



    print(train)

    # test_path = 'test_dataset.tsv'
    # test, _ = to_ids(test_path)
    # # sort_d = test['id'].to_list()
    # # test = sorted(test, is_train=False)
    # # test['text_len'] = list(map(lambda x: len(x), test["content"]))
    # # # 将顺序调回来
    # # test.index = test['id']
    # # test = test.loc[sort_d]
    # test.to_csv('test_data_done.tsv', sep='\t', index=False)

    # train_path = 'train_dataset_v2.tsv'
    # train, dev_df = to_ids2(train_path)
    # # 保存数据
    # train.to_csv('train_data_done.tsv', sep='\t', index=False)
    # dev_df.to_csv('dev_data_done.tsv', sep='\t', index=False)
