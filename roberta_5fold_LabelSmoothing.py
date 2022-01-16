from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
import torch

import os
import gc
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def load_data(data_path, seed_value):
    """载入数据"""
    train = pd.read_csv(data_path + 'train_dataset_v2.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
    test = pd.read_csv(data_path + 'test_dataset.tsv', sep='\t')
    submit = pd.read_csv(data_path + 'submit_example.tsv', sep='\t')
    train = train[train['emotions'].notna()]
    train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN
    test['character'].fillna('无角色', inplace=True)

    train['text'] = train['content'].astype(str) + ' 角色: ' + train['character'].astype(str)
    test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)
    train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
    train_data = train[['text', 'labels']].copy()
    train_data = train_data.sample(frac=1.0, random_state=seed_value)  # 打乱数据
    # train_data_1, dev_df = train_test_split(train_data, test_size=0.2, random_state=seed_value)
    train_df = train_data.copy()
    train_df.reset_index(drop=True, inplace=True)  # 重新排序
    return train_df, test, submit


class BERTDataset(Dataset):
    """自定义数据集"""

    def __init__(self, text, tokenizer, max_len, label=None, is_dev=False):
        self.max_len = max_len
        self.text = text
        self.tokenizer = tokenizer
        self.targets = label
        self.is_dev = is_dev

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        #         token_type_ids = inputs["token_type_ids"]

        if self.is_dev:
            return {
                'ids': torch.LongTensor(ids),
                'mask': torch.LongTensor(mask)
                #                 'token_type_ids': torch.LongTensor(token_type_ids)
            }
        else:
            return {
                'ids': torch.LongTensor(ids),
                'mask': torch.LongTensor(mask),
                #                 'token_type_ids': torch.LongTensor(token_type_ids),
                'targets': torch.FloatTensor(self.targets[index])
            }


class BERTClass(torch.nn.Module):
    """创建模型"""

    def __init__(self, args):
        super(BERTClass, self).__init__()
        self.bert = AutoModel.from_pretrained(args.roberta_path)
        self.l2 = torch.nn.Dropout(args.dropout)
        self.fc = torch.nn.Linear(768, args.num_classes)

    def forward(self, ids, mask):
        _, features = self.bert(ids, attention_mask=mask, return_dict=False)
        features = self.l2(features)
        output = self.fc(features)
        return output


def get_new_labels(y):
    """转换标签"""
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new


def loss_fn(outputs, targets):
    """定义损失函数"""
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def fin(x, threshold):
    if x >= threshold:
        return 1
    return 0


def model_train(args, train_df, tokenizer):
    """模型训练"""
    start_time = time.time()

    train_df["label_new"] = get_new_labels(train_df["labels"])

    # train_dataset = BERTDataset(train_df["text"], tokenizer, args.max_len, train_df["labels"])
    # dev_dataset = BERTDataset(dev_df["text"], tokenizer, args.max_len, dev_df["labels"])

    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True)

    skf = StratifiedKFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed_value)

    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df["text"], train_df["label_new"])):
        print('--------------------------------- {} fold ---------------------------------'.format(i + 1))
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]

        # 顺序标号
        trn_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        train_dataset = BERTDataset(trn_df.text, tokenizer, args.max_len, trn_df.labels)
        dev_dataset = BERTDataset(val_df.text, tokenizer, args.max_len, val_df.labels)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True)

        model = BERTClass(args)
        model.to(args.device)

        # # 普通的优化器
        # optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

        # 分层权重衰减
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.num_warmup_steps,
                                                    num_training_steps=args.epochs * len(train_loader))

        for epoch in range(args.epochs):
            model.train()
            train_targets = []
            train_outputs = []
            for _, data in enumerate(tqdm(train_loader), 0):
                ids = data['ids'].to(args.device)
                mask = data['mask'].to(args.device)
                #         token_type_ids = data['token_type_ids'].to(device)
                targets = data['targets'].to(args.device)
                # 平滑标签
                targets = (1 - args.epsilon) * targets + args.epsilon / args.num_classes

                outputs = model(ids, mask)

                loss = loss_fn(outputs, targets)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5) # 防止梯度爆炸
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_targets.extend(targets.cpu().detach().numpy().tolist())
                train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            #     train_outputs = np.array(train_outputs) >= 0.5
            train_output = [[fin(_, 0.5) for _ in example] for example in train_outputs]
            RMSE = metrics.mean_squared_error(train_targets, train_output, squared=False)
            #     print(f"RMSE Score = {RMSE}")
            print(f"Fin Score = {1 / (1 + RMSE)}")

            model.eval()
            fin_targets = []
            fin_outputs = []
            with torch.no_grad():
                for _, data in enumerate(tqdm(dev_loader), 0):
                    ids = data['ids'].to(args.device)
                    mask = data['mask'].to(args.device)
                    #             token_type_ids = data['token_type_ids'].to(args.device)
                    targets = data['targets'].to(args.device)
                    outputs = model(ids, mask)
                    loss = loss_fn(outputs, targets)
                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            #     fin_outputs = np.array(fin_outputs) >= 0.5
            fin_target = [[fin(_, 1.0) for _ in example] for example in fin_targets]
            fin_output = [[fin(_, 0.5) for _ in example] for example in fin_outputs]
            RMSE = metrics.mean_squared_error(fin_targets, fin_output, squared=False)
            accuracy = metrics.accuracy_score(fin_target, fin_output)
            #     print(f"RMSE Score = {RMSE}")
            print(f"Fin Score = {1 / (1 + RMSE)}")
            print(f"Accuracy Score = {accuracy}")

    torch.save(model.state_dict(), args.save_path + args.model_name + '.bin')


def model_predict(args, test, submit, tokenizer):
    test_dataset = BERTDataset(test["text"], tokenizer, args.max_len, is_dev=True)
    test_loader = DataLoader(test_dataset, batch_size=args.valid_batch_size)

    test_model = BERTClass(args)
    test_model.to(args.device)
    test_model.load_state_dict(torch.load(args.save_path + args.model_name + '.bin'))
    test_model.eval()
    preds = None
    pred_end = None
    preds = np.empty((len(test_dataset), 6))
    pred_end = np.empty((len(test_dataset), 6))
    n_batches = len(test_loader)
    for step, batch in enumerate(tqdm(test_loader), 0):
        ids = batch['ids'].to(args.device)
        mask = batch['mask'].to(args.device)
        #     token_type_ids = batch['token_type_ids'].to(device)
        with torch.no_grad():
            outputs = test_model(ids, mask)
            pred_probs = torch.sigmoid(outputs)
        start_index = args.valid_batch_size * step
        end_index = (start_index + args.valid_batch_size if step != (n_batches - 1) else len(test_dataset))
        preds[start_index:end_index] = pred_probs.cpu().detach().numpy()

    pred_end = [[fin(_, 0.5) for _ in example] for example in preds]

    sub = submit.copy()
    sub['emotion'] = pred_end
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

    sub.to_csv(args.results_path + args.model_name + '.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='roberta_5fold_LabelSmoothing', type=str)
    parser.add_argument('--data_path', default='./raw_data/', type=str)
    parser.add_argument('--roberta_path', default='./pre_models/chinese-roberta-wwm-ext/', type=str)
    # parser.add_argument('--roberta_path', default='hfl/chinese-roberta-wwm-ext', type=str) # 运行时自动下载
    parser.add_argument('--save_path', default='./save_model/', type=str)
    parser.add_argument('--results_path', default='./results/', type=str)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--seed_value', default=2021, type=int)
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--valid_batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--num_warmup_steps', default=50, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    args = parser.parse_args(args=[])

    start_time = time.time()
    # 设置随机数种子
    setup_seed(args.seed_value)

    # 载入数据
    print('loading data ...')
    train_df, test, submit = load_data(args.data_path, args.seed_value)

    tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)

    # 训练
    print('start training ...')
    model_train(args, train_df, tokenizer)

    # 预测
    print('start predicting...')
    model_predict(args, test, submit, tokenizer)
