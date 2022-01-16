from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch import nn
import torch

import os
import gc
import re
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import itertools

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
        text_df[i] = re.sub(r'[{}]+'.format(punctuation), '', text_raw['text'][i])
    text_raw['text'] = text_df
    return text_raw


def load_data(data_path, seed_value):
    """载入数据"""
    train = pd.read_csv(data_path + 'train_dataset_v2.tsv', sep='\t', error_bad_lines=False, warn_bad_lines=False)
    test = pd.read_csv(data_path + 'test_dataset.tsv', sep='\t')
    submit = pd.read_csv(data_path + 'submit_example.tsv', sep='\t')
    train = train[train['emotions'].notna()]
    train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN
    test['character'].fillna('无角色', inplace=True)

    # train['text'] = train['content'].astype(str) + ' 角色: ' + train['character'].astype(str)
    # test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)
    # train['text'] = '（描述角色：' + train['character'].astype(str) + ' ）' + train['content'].astype(str)
    # test['text'] = '（描述角色：' + test['character'].astype(str) + ' ）' + test['content'].astype(str)
    train['text'] = train['character'].astype(str) + train['content'].astype(str)
    test['text'] = test['character'].astype(str) + test['content'].astype(str)
    train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
    train_data = train[['text', 'labels']].copy()
    train_data = train_data.sample(frac=1.0, random_state=seed_value)  # 打乱数据
    # train_data_1, dev_df = train_test_split(train_data, test_size=0.2, random_state=seed_value)
    train_df = train_data.copy()
    train_df.reset_index(drop=True, inplace=True)  # 重新排序

    # 去除标点符号
    # train_df = remove_punctuation(train_df)
    # test = remove_punctuation(test)

    return train_df, test, submit


class convert_input(object):
    def __init__(self, bert_path):
        self.tips = "deal_data"
        self.bert_path = bert_path

    def convert_inputs(self, text, max_seq_length, target=None, is_dev=False):
        """为每一个句子标签生成对应的input_ids，attention_mask，token_type_ids"""
        token_label_list = ['UNLABEL', 'LABEL']
        token_label_map = {}
        for (i, label) in enumerate(token_label_list):
            token_label_map[label] = i
        # 6标签
        label_list = ['label0', 'label1', 'label2', 'label3', 'label4', 'label5']
        # 24标签
        # label_list = ['label0', 'label1', 'label2', 'label3', 'label4', 'label5',
        #               'label6', 'label7', 'label8', 'label9', 'label10', 'label11',
        #               'label12', 'label13', 'label14', 'label15', 'label16', 'label17',
        #               'label18', 'label19', 'label20', 'label21', 'label22', 'label23']
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = [y for y in text]

        bias = 1
        label_trans_token = {}
        for (i, label) in enumerate(label_list):
            if i + bias < 100:
                label_trans_token[label] = i + bias
            else:
                label_trans_token[label] = i + bias + 4

        if not is_dev:
            label_li = []
            for i in range(len(target)):
                if (target[i] == 1):
                    label_li.append(label_list[i])
            label_list = label_li
        else:
            label_list = []

            # label_ids = target

        # right_labels = []
        # wrong_labels = []
        # for label_id in range(0, len(label_ids)):
        #     if label_ids[label_id] == 1:
        #         right_labels.append(label_id)
        #     else:
        #         wrong_labels.append(label_id)

        # right_pair = list(itertools.combinations(right_labels, 2))
        #
        # contrast_dict = {}
        # for pair in right_pair:
        #     contrast_dict[pair] = [0, 1]
        # for i in range(0, int(len(right_pair) * 2)):
        #     r = random.sample(right_labels, 1)[0]
        #     w = random.sample(wrong_labels, 1)[0]
        #     contrast_dict[(r, w)] = [1, 0]

        def _general_token_b_and_seq_label(predicate_label, label_trans_token):
            token_b = []
            token_b_ids = []
            seq_label_token = []
            for k, v in label_trans_token.items():
                token_b.append(k)
                token_b_ids.append(v)
                if k not in predicate_label:
                    seq_label_token.append("UNLABEL")
                else:
                    seq_label_token.append("LABEL")
            return token_b, token_b_ids, seq_label_token

        token_b, token_b_ids, token_b_label = _general_token_b_and_seq_label(label_list, label_trans_token)

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                ##cut sentence frist deal with tokens_a
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        if token_b_ids:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, token_b_ids, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        token_label_ids = []
        fit_labelspace_positions = []
        fit_docspace_positions = []
        doc_idx = 0

        tokens.append("[CLS]")
        segment_ids.append(0)
        token_label_ids.append(token_label_map["UNLABEL"])
        fit_docspace_positions.append(doc_idx)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
            token_label_ids.append(token_label_map["UNLABEL"])
            doc_idx += 1
            fit_docspace_positions.append(doc_idx)

        tokens.append("[SEP]")
        segment_ids.append(0)
        token_label_ids.append(token_label_map["UNLABEL"])
        doc_idx += 1
        fit_docspace_positions.append(doc_idx)

        tokenizers = BertTokenizer.from_pretrained(self.bert_path)
        input_ids = tokenizers.convert_tokens_to_ids(tokens)
        lsp_idx = len(input_ids)

        if token_b_ids:
            for tk, tbi, tbl in zip(token_b, token_b_ids, token_b_label):
                tokens.append(tk)
                input_ids.append(tbi)
                segment_ids.append(1)
                token_label_ids.append(token_label_map[tbl])
                fit_labelspace_positions.append(lsp_idx)
                lsp_idx += 1

        tokens.append("[SEP]")
        input_ids.append(tokenizers.convert_tokens_to_ids(["[SEP]"])[0])
        segment_ids.append(0)
        token_label_ids.append(token_label_map["UNLABEL"])
        doc_idx = lsp_idx
        fit_docspace_positions.append(doc_idx)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            tokens.append("[Padding]")
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_label_ids.append(token_label_map["UNLABEL"])
            doc_idx += 1
            fit_docspace_positions.append(doc_idx)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_label_ids) == max_seq_length
        assert (len(fit_docspace_positions) + len(fit_labelspace_positions)) == max_seq_length

        if not is_dev:
            return {'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'token_type_ids': segment_ids,
                    'token_label_ids': token_label_ids,
                    'fit_docspace_positions': fit_docspace_positions,
                    'fit_labelspace_positions': fit_labelspace_positions,
                    'target_label': target
                    }
        else:
            return {'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'token_type_ids': segment_ids,
                    'token_label_ids': token_label_ids,
                    'fit_docspace_positions': fit_docspace_positions,
                    'fit_labelspace_positions': fit_labelspace_positions
                    }


class BERTDataset(Dataset):
    """自定义数据集"""

    def __init__(self, text, max_len, is_dev=False):
        self.max_len = max_len
        self.text = text
        self.is_dev = is_dev

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]

        ids = text['input_ids']
        mask = text['attention_mask']
        token_type_ids = text["token_type_ids"]
        fit_docspace_positions = text['fit_docspace_positions']
        fit_labelspace_positions = text['fit_labelspace_positions']

        if self.is_dev:
            return {
                'ids': torch.LongTensor(ids),
                'mask': torch.LongTensor(mask),
                'token_type_ids': torch.LongTensor(token_type_ids),
                'fit_docspace_positions': torch.LongTensor(fit_docspace_positions),
                'fit_labelspace_positions': torch.LongTensor(fit_labelspace_positions)
            }
        else:
            return {
                'ids': torch.LongTensor(ids),
                'mask': torch.LongTensor(mask),
                'token_type_ids': torch.LongTensor(token_type_ids),
                'targets': torch.FloatTensor(text['target_label']),
                'fit_docspace_positions': torch.LongTensor(fit_docspace_positions),
                'fit_labelspace_positions': torch.LongTensor(fit_labelspace_positions)
            }


class BERTClass(torch.nn.Module):
    """创建模型"""

    def __init__(self, args):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.num_classes = args.num_classes
        self.args = args

        token_label_output_weight = nn.Parameter(torch.randn(args.num_classes, 768))
        self.token_label_output_weight = truncated_normal_(token_label_output_weight, std=0.02)  # 使用截断正态分布来初始化
        self.token_label_output_bias = nn.Parameter(torch.randn(args.num_classes))
        nn.init.constant_(self.token_label_output_bias, 0)

        self.joint_embedding = joint_embedding(args)
        self.l2 = nn.Dropout(args.dropout)
        # self.leakyrulu = nn.LeakyReLU()
        self.fc = nn.Linear(768, args.num_classes)

    def forward(self, ids, mask, token_type_ids, label_positions, doc_positions):
        _, features = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        # _[32,110,768]
        # _, features = self.bert(input_ids=ids, attention_mask=mask, return_dict=False)
        # features = self.l2(features)
        # features = self.leakyrulu(features)
        token_label_hidden_size = _.shape[-1]  # [768]
        doc_seq_length = _.shape[-2] - self.num_classes  # [104]
        doc_output_layer = gather_indexes(_, doc_positions, self.args)
        doc_output_layer = torch.reshape(doc_output_layer, [-1, doc_seq_length, token_label_hidden_size])

        token_label_output_layer = gather_indexes(_, label_positions, self.args)
        token_label_output_layer = torch.reshape(token_label_output_layer,
                                                 [-1, self.num_classes, token_label_hidden_size])

        H_enc = self.joint_embedding(doc_output_layer, token_label_output_layer)

        H_enc = self.l2(H_enc)
        logits_wx = torch.matmul(H_enc, self.token_label_output_weight.permute(1, 0))
        logits = logits_wx + self.token_label_output_bias
        return logits


class joint_embedding(torch.nn.Module):
    """创建模型"""

    def __init__(self, args):
        super(joint_embedding, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, args.num_classes, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, doc_output_layer, token_label_output_layer):
        x_emb_norm = F.normalize(doc_output_layer, dim=2, p=2)
        w_class_norm = F.normalize(token_label_output_layer, dim=2, p=2)
        w_class_norm = w_class_norm.permute(0, 2, 1)
        G = torch.matmul(x_emb_norm, w_class_norm)
        G = torch.unsqueeze(G, axis=-1)
        label_length = token_label_output_layer.shape[-2]
        G = G.permute(0, 3, 1, 2)
        Att_v = self.conv2d(G)
        Att_v = Att_v.permute(0, 2, 3, 1)
        Att_v = torch.max(Att_v, dim=-1, keepdims=True)[0]
        Att_v = torch.squeeze(Att_v)
        Att_v = torch.max(Att_v, dim=-1, keepdims=True)[0]
        Att_v_tan = torch.tanh(Att_v)
        x_emb_norm = torch.squeeze(x_emb_norm)
        x_att = x_emb_norm * Att_v_tan
        H_enc = torch.sum(x_att, axis=1)
        # two_w_class_norm = gather_indexes(token_label_output_layer, pair)#32,2,768
        # two_w_class_norm = torch.reshape(two_w_class_norm,[-1,2,768])
        # two_w_class_norm = torch.transpose(two_w_class_norm , [0, 2, 1])#32,768,2
        # G_pair_a = two_w_class_norm[:,:,0]
        # G_pair_a = torch.reshape(G_pair_a,[-1,768])
        # G_pair_b = two_w_class_norm[:, :, 1]
        # G_pair_b = torch.reshape(G_pair_b, [-1, 768])
        # a_b_enc = torch.concat([G_pair_a , G_pair_b],-1)
        return H_enc


def truncated_normal_(tensor, mean=0, std=0.02):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def gather_indexes(sequence_tensor, positions, args):
    """Gathers the vectors at the specific positions over a minibatch."""
    batch_size = sequence_tensor.shape[0]
    seq_length = sequence_tensor.shape[1]
    width = sequence_tensor.shape[2]

    flat_offsets = torch.reshape(
        torch.arange(0, batch_size) * seq_length, [-1, 1]).to(args.device)
    flat_positions = torch.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = torch.reshape(sequence_tensor, [batch_size * seq_length, width])
    flat_positions = flat_positions.unsqueeze(1).expand(1, flat_positions.shape[0],
                                                        flat_sequence_tensor.shape[1]).squeeze()
    output_tensor = torch.gather(flat_sequence_tensor, 0, flat_positions)
    return output_tensor


def get_new_labels(y):
    """转换标签"""
    y_new = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_new


def mse_loss_fn(outputs, targets):
    """定义MSE损失函数"""
    return torch.nn.MSELoss()(outputs, targets)


def fin(x):
    """转化标签,标签取整"""
    if x < 0:
        return 0
    elif x > 3:
        return 3
    else:
        return x


#     if x < 0.5:
#         return 0
#     elif x >= 0.5 and x < 1.5:
#         return 1
#     elif x >= 1.5 and x < 2.5:
#         return 2
#     else:
#         return 3


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

def model_train(args, train_df):
    """模型训练"""
    start_time = time.time()

    train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=args.seed_value)

    train_dataset = BERTDataset(train_df, args.max_len)
    dev_dataset = BERTDataset(dev_df, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True)

    model = BERTClass(args)
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
    # 学习率的调整
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
            ids = data['ids'].to(args.device)
            mask = data['mask'].to(args.device)
            token_type_ids = data['token_type_ids'].to(args.device)
            targets = data['targets'].to(args.device)
            label_positions = data['fit_labelspace_positions'].to(args.device)
            doc_positions = data['fit_docspace_positions'].to(args.device)
            # # 平滑标签
            # targets = (1 - args.epsilon) * targets + args.epsilon / args.num_classes
            optimizer.zero_grad()
            outputs = model(ids, mask, token_type_ids, label_positions, doc_positions)

            loss = mse_loss_fn(outputs, targets)

            loss.backward()
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            outputs = model(ids, mask, token_type_ids, label_positions, doc_positions)
            loss_adv = mse_loss_fn(outputs, targets)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
            # 梯度下降，更新参数
            nn.utils.clip_grad_norm_(model.parameters(), 5)  # 防止梯度爆炸
            optimizer.step()
            scheduler.step()

            # 处理loss
            trn_loss += loss.item()

        print(f'[train]  Epoch: {epoch}, Loss:  {trn_loss / len(train_loader)}')

        model.eval()
        dev_loss = 0
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(tqdm(dev_loader), 0):
                ids = data['ids'].to(args.device)
                mask = data['mask'].to(args.device)
                token_type_ids = data['token_type_ids'].to(args.device)
                targets = data['targets'].to(args.device)
                label_positions = data['fit_labelspace_positions'].to(args.device)
                doc_positions = data['fit_docspace_positions'].to(args.device)
                outputs = model(ids, mask, token_type_ids, label_positions, doc_positions)
                loss = mse_loss_fn(outputs, targets)

                # 处理loss
                dev_loss += loss.item()

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            print(f'[eval]  Epoch: {epoch}, Loss:  {dev_loss / len(dev_loader)}')
        #     fin_outputs = np.array(fin_outputs) >= 0.5
        fin_output = [[fin(_) for _ in example] for example in fin_outputs]
        RMSE = metrics.mean_squared_error(fin_targets, fin_output, squared=False)
        print(f"Fin Score = {1 / (1 + RMSE)}")

        # scheduler.step()
        earlystop(dev_loss / len(dev_loader), model, epoch)
        if earlystop.early_stop:
            break

    # torch.save(model.state_dict(), args.save_path + args.model_name + '.bin')
    end_time = time.time()
    print("tra_all_time:", end_time - start_time)


def model_predict(args, test, submit):
    test_dataset = BERTDataset(test, args.max_len, is_dev=True)
    test_loader = DataLoader(test_dataset, batch_size=args.valid_batch_size)

    test_model = BERTClass(args)
    test_model.to(args.device)
    test_model.load_state_dict(torch.load(args.save_path + args.model_name + '.bin'))
    test_model.eval()

    preds = np.empty((len(test_dataset), args.num_classes))

    n_batches = len(test_loader)
    for step, batch in enumerate(tqdm(test_loader), 0):
        ids = batch['ids'].to(args.device)
        mask = batch['mask'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        label_positions = batch['fit_labelspace_positions'].to(args.device)
        doc_positions = batch['fit_docspace_positions'].to(args.device)
        with torch.no_grad():
            outputs = test_model(ids, mask, token_type_ids, label_positions, doc_positions)
            pred_probs = outputs.cpu().detach().numpy()
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
    parser.add_argument('--model_name', default='regression_LACO_fgm', type=str)
    parser.add_argument('--data_path', default='./raw_data/', type=str)
    parser.add_argument('--bert_path', default='./pre_models/chinese-roberta-wwm-ext/', type=str)
    # parser.add_argument('--bert_path', default='bert-base-chinese', type=str) # 运行时自动下载
    parser.add_argument('--save_path', default='./save_model/', type=str)
    parser.add_argument('--results_path', default='./results/', type=str)
    # parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--seed_value', default=2021, type=int)
    parser.add_argument('--max_len', default=110, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--valid_batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    # parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--epsilon', default=0.1, type=float)  # 平滑标签参数
    parser.add_argument('--num_warmup_steps', default=0, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    args = parser.parse_args(args=[])

    start_time = time.time()
    # 设置随机数种子
    setup_seed(args.seed_value)

    # 载入数据
    print('loading data ...')
    train_df, test, submit = load_data(args.data_path, args.seed_value)
    tokenizer = convert_input(args.bert_path)
    train_data = []
    test_data = []
    if not os.path.exists(args.data_path + 'LACO_train_data_raw_des_persona_text.npy'):
        for i in range(len(train_df)):
            print(i)
            train_da = tokenizer.convert_inputs(train_df['text'].to_list()[i], args.max_len,
                                                train_df['labels'].to_list()[i])
            train_data.append(train_da)
        print('保存训练数据...')
        np.save(args.data_path + 'LACO_train_data_raw_des_persona_text.npy', train_data)
    else:
        print('载入训练数据...')
        train_data = np.load(args.data_path + 'LACO_train_data_raw_des_persona_text.npy', allow_pickle=True)
    if not os.path.exists(args.data_path + 'LACO_test_data_raw_des_persona_text.npy'):
        for i in range(len(test)):
            print(i)
            test_da = tokenizer.convert_inputs(test['text'].to_list()[i], args.max_len, is_dev=True)
            test_data.append(test_da)
        print('保存训练数据...')
        np.save(args.data_path + 'LACO_test_data_raw_des_persona_text.npy', test_data)
    else:
        print('载入测试数据...')
        test_data = np.load(args.data_path + 'LACO_test_data_raw_des_persona_text.npy', allow_pickle=True)
    print('loading data done...')

    # 训练
    print('start training ...')
    model_train(args, train_data)

    # 预测
    print('start predicting...')
    model_predict(args, test_data, submit)

    """
    为了避免载入数据太慢，这里现将数据加载好保存下来，再载入
    

    """