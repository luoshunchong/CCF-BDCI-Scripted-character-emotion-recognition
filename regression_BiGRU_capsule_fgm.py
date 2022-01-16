from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch import nn
import torch
from torch.autograd import Variable

import os
import gc
import re
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import random

warnings.filterwarnings('ignore')

# 分数68


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    cudnn.benchmark = False

def remove_punctuation(text_raw):
    """去除中文标点符号"""
    punctuation = "。: ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    text_df = [[] for _ in range(len(text_raw['text']))]
    for i in range(len(text_raw['text'])):
        text_df[i] = re.sub(r'[{}]+'.format(punctuation),'',text_raw['text'][i])
    text_raw['text'] = text_df
    return text_raw


def load_data(data_path, seed_value):
    """载入数据"""
    train = pd.read_csv(data_path + 'train_data_done.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
    test = pd.read_csv(data_path + 'test_data_done.tsv', sep='\t')
    submit = pd.read_csv(data_path + 'submit_example.tsv', sep='\t')
#     train = train[train['emotions'].notna()]
#     train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN
#     test['character'].fillna('无角色', inplace=True)
#     # 提取剧本id,场次id
#     train['id_t'] = train['id'].apply(lambda x: [i for i in x.split('_')])
#     train['id_t'] = train['id_t'].apply(lambda x: [i for i in x[:-1]])
#     train['id_t'] = train['id_t'].apply(lambda x: '_'.join(x))
#     # 提取剧本id,场次id
#     test['id_t'] = test['id'].apply(lambda x: [i for i in x.split('_')])
#     test['id_t'] = test['id_t'].apply(lambda x: [i for i in x[:-1]])
#     test['id_t'] = test['id_t'].apply(lambda x: '_'.join(x))

    # train['text'] = train['content'].astype(str) + ' 角色: ' + train['character'].astype(str)
    # test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)
    # train['text'] = '（描述角色：' + train['character'].astype(str) + ' ）' + train['content'].astype(str)
    # test['text'] = '（描述角色：' + test['character'].astype(str) + ' ）' + test['content'].astype(str)
    train['text'] = train['id_t'] + ':' + train['character'].astype(str) + ':' + train['content'].astype(str)
    test['text'] = test['id_t'] + ':' + test['character'].astype(str) + ':' + test['content'].astype(str)
    train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
    train_data = train[['text', 'labels']].copy()
    train_data = train_data.sample(frac=1.0, random_state=seed_value)  # 打乱数据
    train_df = train_data.copy()
    train_df.reset_index(drop=True, inplace=True)  # 重新排序

    # 去除标点符号
    # train_df = remove_punctuation(train_df)
    # test = remove_punctuation(test)

    return train_df, test, submit


class EarlyStopping:
    """早停"""

    def __init__(self, early_stop_round, model_path):
        self.early_stop_round = early_stop_round
        self.model_path = model_path

        self.counter = 0
        self.best_loss = float('inf')  # 表示正无穷
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.early_stop_round:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_loss = val_loss
            print("保存第" + str(epoch) + "次模型参数！")
            torch.save(model.state_dict(), self.model_path)

def load_embeddings(args,train_df):
    tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    vocab_dic = {}
    for line in train_df['text']:
        content = line.strip()
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= 1], key=lambda x: x[1], reverse=True)[
                 :10000]
    word_to_id = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    word_to_id.update({'<UNK>': len(word_to_id), '<PAD>': len(word_to_id) + 1})
    embeddings = np.random.rand(len(word_to_id), args.emb_dim)
    f = open(args.w2v_path+'sgns.merge.bigram', "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    return embeddings, word_to_id

def load_dataset(text, pad_size, vocab, is_train=True):
    tokenizer = lambda x: [y for y in x]  # char-level
    contents = []
    for line in text['text']:
        content = line
        words_line = []
        token = tokenizer(content)
        if pad_size:
            if len(token) < pad_size:
                token.extend(['<PAD>'] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get('<UNK>')))
        contents.append(words_line)
    contents = np.array(contents)
    if is_train:
        contents_label = np.array(text['labels'].values.tolist())
        return contents, contents_label
    else:
        return contents


class Model(nn.Module):
    def __init__(self, config, embeddings):
        super(Model, self).__init__()
        # embedding
        embeddings = torch.from_numpy(embeddings).float()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)

        # GRU
        self.BiGRU = nn.GRU(config.emb_dim, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True,
                            dropout=config.dropout)

        # Primary Layer
        self.primary_capsules_doc = PrimaryCaps(num_capsules=config.dim_capsule, in_channels=config.max_len, out_channels=32,
                                                kernel_size=1, stride=1)
        # FlattenCaps
        self.flatten_capsules = FlattenCaps()
        # W_doc初始化
        self.W_doc = nn.Parameter(torch.FloatTensor(9600, config.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)
        # FCCaps
        self.fc_capsules_doc_child = FCCaps(config, output_capsule_num=config.num_classes,
                                            input_capsule_num=config.num_compressed_capsule,
                                            in_channels=config.dim_capsule, out_channels=config.dim_capsule)

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0, 2, 1), W).permute(0, 2, 1)
        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def forward(self, x):  # shape[128,32]
        content1 = self.embedding(x)  # shape[128,32,300]

        nets_doc, _ = self.BiGRU(content1)  # [128,32,256]

        poses_doc, activations_doc = self.primary_capsules_doc(
            nets_doc)  # poses_doc[128,16,32,446,1],activations_doc[128,32,446,1]
        poses, activations = self.flatten_capsules(poses_doc,
                                                   activations_doc)  # poses[128,14272,16],activations[128,14272,1]
        poses, activations = self.compression(poses, self.W_doc)  # poses[128,128,16],activations[128,128]
        poses, activations = self.fc_capsules_doc_child(poses, activations)
        # poses[128,10,16,1],activations[128,10,1]
        activations = activations.squeeze(2)
        return activations


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)

        torch.nn.init.xavier_uniform_(self.capsules.weight)

        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        poses = squash_v1(u, axis=1)
        activations = torch.sqrt((poses ** 2).sum(1))
        return poses, activations


class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()

    def forward(self, p, a):
        poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)  # [64,14272,16]
        activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)  # [64,14272,1]
        return poses, activations

def Adaptive_KDE_routing(batch_size, b_ij, u_hat):
    last_loss = 0.0
    while True:
        if False:
            leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
            leaky_logits = torch.cat((leak, b_ij), 2)
            leaky_routing = F.softmax(leaky_logits, dim=2)
            c_ij = leaky_routing[:, :, 1:, :].unsqueeze(4)
        else:
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
        c_ij = c_ij / c_ij.sum(dim=1, keepdim=True)
        v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
        dd = 1 - ((squash_v1(u_hat, axis=3) - v_j) ** 2).sum(3)
        b_ij = b_ij + dd

        c_ij = c_ij.view(batch_size, c_ij.size(1), c_ij.size(2))
        dd = dd.view(batch_size, dd.size(1), dd.size(2))

        kde_loss = torch.mul(c_ij, dd).sum() / batch_size
        kde_loss = np.log(kde_loss.item())

        if abs(kde_loss - last_loss) < 0.05:
            break
        else:
            last_loss = kde_loss
    poses = v_j.squeeze(1)
    activations = torch.sqrt((poses ** 2).sum(2))
    return poses, activations


def squash_v1(x, axis):
    s_squared_norm = (x ** 2).sum(axis, keepdim=True)  # 按行相加，并且保持其二维特性[64,1,32,446,1]
    scale = torch.sqrt(s_squared_norm) / (0.5 + s_squared_norm)  # [64,1,32,446,1]
    return scale * x


class FCCaps(nn.Module):
    def __init__(self, args, output_capsule_num, input_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num

        self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, out_channels,
                                                 in_channels))  # [1,128,3954,16,16]
        torch.nn.init.xavier_uniform_(self.W1)

        self.device = args.device
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size = x.size(0)
        variable_output_capsule_num = 6
        W1 = self.W1[:, :, [0,1,2,3,4,5], :, :]  # [1,128,276,16,16]

        x = torch.stack([x] * variable_output_capsule_num, dim=2).unsqueeze(4)  # [64,128,276,16,1]

        W1 = W1.repeat(batch_size, 1, 1, 1, 1)  # [64,128,276,16,16]
        u_hat = torch.matmul(W1, x)

        b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, variable_output_capsule_num, 1)).to(self.device)

        poses, activations = Adaptive_KDE_routing(batch_size, b_ij, u_hat)
        return poses, activations



def get_new_labels(y):
    """转换标签"""
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new

def mse_loss_fn(outputs, targets):
    """定义MSE损失函数"""
    return torch.nn.MSELoss()(outputs, targets)

def fin(x):
    """转化标签,标签取整"""
    # if x < 0.5:
    #     return 0
    # elif x >= 0.5 and x < 1.5:
    #     return 1
    # elif x >= 1.5 and x < 2.5:
    #     return 2
    # else:
    #     return 3
    if x < 0:
        return 0
    elif x > 3:
        return 3
    else:
        return x

class FGM():
    """对抗学习fgm"""
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def model_train(args, train_df, embeddings, vocab):
    """模型训练"""
    start_time = time.time()

    train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=args.seed_value)
    train_df.reset_index(drop=True, inplace=True)  # 重新排序
    dev_df.reset_index(drop=True, inplace=True)  # 重新排序


    train, train_label = load_dataset(train_df, args.max_len, vocab)
    dev, dev_label = load_dataset(dev_df, args.max_len, vocab)


    train_dataset = TensorDataset(torch.from_numpy(train).type(torch.LongTensor),
                                    torch.from_numpy(train_label).type(torch.FloatTensor))
    dev_dataset = TensorDataset(torch.from_numpy(dev).type(torch.LongTensor),
                                    torch.from_numpy(dev_label).type(torch.FloatTensor))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True)

    model = Model(args, embeddings)
    model.to(args.device)
    # 初始化
    fgm = FGM(model)

    # # 普通的优化器
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    # 分层权重衰减
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0},
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=args.epochs * len(train_loader))

    # 余弦退火学习率
    # scheduler = CosineAnnealingLR(optimizer, args.epochs)
    # 早停策略
    earlystop = EarlyStopping(early_stop_round=2, model_path=args.save_path + args.model_name + '.bin')

    for epoch in range(args.epochs):
        model.train()
        trn_loss = 0
        for _, data in enumerate(tqdm(train_loader), 0):
            ids = data[0].to(args.device)
            targets = data[1].to(args.device)
            # # 平滑标签
            # targets = (1 - args.epsilon) * targets + args.epsilon / args.num_classes
            outputs = model(ids)

            loss = mse_loss_fn(outputs, targets)

            loss.backward()
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            outputs = model(ids)
            loss_adv = mse_loss_fn(outputs, targets)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
            # 梯度下降，更新参数
#             nn.utils.clip_grad_norm_(model.parameters(), 5)  # 防止梯度爆炸
            optimizer.step()
            # 将梯度清零
            model.zero_grad()
            scheduler.step()

            # 处理loss
            trn_loss += loss.item()

        print(f'[train]  Epoch: {epoch}, Loss:  {trn_loss/len(train_loader)}')

        model.eval()
        dev_loss = 0
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(tqdm(dev_loader), 0):
                ids = data[0].to(args.device)
                targets = data[1].to(args.device)
                outputs = model(ids)

                loss = mse_loss_fn(outputs, targets)

                # 处理loss
                dev_loss += loss.item()

                fin_targets.extend(targets.cpu().numpy().tolist())
                fin_outputs.extend(outputs.cpu().numpy().tolist())
            print(f'[eval]  Epoch: {epoch}, Loss:  {dev_loss/len(dev_loader)}')
        #     fin_outputs = np.array(fin_outputs) >= 0.5
        fin_output = [[fin(_) for _ in example] for example in fin_outputs]
        RMSE = metrics.mean_squared_error(fin_targets, fin_output, squared=False)
        print(f"Fin Score = {1 / (1 + RMSE)}")

        # scheduler.step()
        earlystop(dev_loss/len(dev_loader), model, epoch)
        if earlystop.early_stop:
            break

    # torch.save(model.state_dict(), args.save_path + args.model_name + '.bin')
    end_time = time.time()
    print("tra_all_time:", end_time-start_time)



def model_predict(args, test, embeddings, submit, vocab):
    test = load_dataset(test, args.max_len, vocab, is_train=False)

    test_dataset = TensorDataset(torch.from_numpy(test).type(torch.LongTensor))
    test_loader = DataLoader(test_dataset, batch_size=args.valid_batch_size)

    test_model = Model(args, embeddings)
    test_model.to(args.device)
    test_model.load_state_dict(torch.load(args.save_path + args.model_name + '.bin'))
    test_model.eval()

    preds = np.empty((len(test_dataset), args.num_classes))

    n_batches = len(test_loader)
    for step, batch in enumerate(tqdm(test_loader), 0):
        ids = batch[0].to(args.device)
        with torch.no_grad():
            outputs = test_model(ids)
            pred_probs = outputs.cpu().numpy()
        start_index = args.valid_batch_size * step
        end_index = (start_index + args.valid_batch_size if step != (n_batches - 1) else len(test_dataset))
        preds[start_index:end_index] = pred_probs

    pred_end = [[fin(_) for _ in example] for example in preds]
    sub = submit.copy()
    sub['emotion'] = pred_end
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

    sub.to_csv(args.results_path + args.model_name + '.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='regression_BiGRU_capsule_fgm', type=str)
    parser.add_argument('--data_path', default='./raw_data/', type=str)
    parser.add_argument('--w2v_path', default='./w2v_chinese/', type=str)
    parser.add_argument('--save_path', default='./save_model/', type=str)
    parser.add_argument('--results_path', default='./results/', type=str)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--seed_value', default=2021, type=int)
    parser.add_argument('--max_len', default=128, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--valid_batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--num_warmup_steps', default=50, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--num_compressed_capsule', default=128, type=int)
    parser.add_argument('--dim_capsule', default=16, type=int)
    parser.add_argument('--dim_model', default=300, type=int)
    parser.add_argument('--num_head', default=6, type=int)
    parser.add_argument('--hidden_size', default=150, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    args = parser.parse_args(args=[])

    start_time = time.time()
    # 设置随机数种子
    setup_seed(args.seed_value)

    # 载入数据
    print('loading data ...')
    train_df, test, submit = load_data(args.data_path, args.seed_value)
    if not os.path.exists(args.w2v_path + 'sgns.merge.bigram.npy'):
        embeddings, vocab = load_embeddings(args, train_df)
        np.save(args.w2v_path + 'sgns.merge.bigram.npy', embeddings)
        pkl.dump(vocab, open(args.w2v_path + 'vocab.pkl', 'wb'))
    else:
        embeddings = np.load(args.w2v_path + 'sgns.merge.bigram.npy', allow_pickle=True)
        vocab = pkl.load(open(args.w2v_path + 'vocab.pkl', 'rb'))

    # 训练
    print('start training ...')
    model_train(args, train_df, embeddings, vocab)

    # 预测
    print('start predicting...')
    model_predict(args, test, embeddings, submit, vocab)

"""

"""