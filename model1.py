import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler,TensorDataset
from transformers import BertTokenizer, AutoTokenizer, BertModel, BertConfig, AutoModel, AdamW
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

"""
只使用'hfl/chinese-roberta-wwm-ext'预训练模型，1/(1+RMSE)分数0.68
"""

seed_value = 2021
# 设置随机数
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    cudnn.benchmark = False
# 设置随机数种子
setup_seed(seed_value)

# 数据预处理
train = pd.read_csv('raw_data/train_dataset_v2.tsv', sep='\t',
                    error_bad_lines=False, warn_bad_lines=False)
test = pd.read_csv('raw_data/test_dataset.tsv', sep='\t')
submit = pd.read_csv('raw_data/submit_example.tsv', sep='\t')
train = train[train['emotions'].notna()]
train['character'].fillna('无角色', inplace=True)  # 使用‘无角色’填充NA/NaN
test['character'].fillna('无角色', inplace=True)
train['text'] = train['content'].astype(str) + ' 角色: ' + train['character'].astype(str)

#  查看句子的平均长度，设置MAX_LEN
train['text_len'] = list(map(lambda x:len(x),train["text"]))
train.head()

# #这个时候 再用上面的方法 绘制一下句子长度的分布图
# sns.countplot("text_len",data=train)
# plt.title("the text_len in train")
# plt.xticks([])#这里把x轴上面的坐标值设置为空 否则显示太密集  一坨黑的
# plt.show()#长度的值不像label只有 两个  这种可视化方式好像不太合适
# #横轴是句子的长度  纵轴是对应长度的句子的数量
# #用柱状图试试
# sns.distplot(train["text_len"])
# plt.show()#横轴是句子的长度 纵轴是概率密度

test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)
train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
train_data = train[['text', 'labels']].copy()
# train_data = train_data.sample(frac=1.0, random_state=42)  # 打乱数据
train_data_1, dev_df = train_test_split(train_data, test_size=0.2, random_state=seed_value)
train_df = train_data_1.copy()
train_df.reset_index(drop=True, inplace=True)# 重新排序
dev_df.reset_index(drop=True, inplace=True)# 重新排序

# 参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 100
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 2e-5
pre_model = 'hfl/chinese-roberta-wwm-ext'
tokenizer = AutoTokenizer.from_pretrained(pre_model)


class BERTDataset(Dataset):
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


train_dataset = BERTDataset(train_df["text"], tokenizer, MAX_LEN, train_df["labels"])
dev_dataset = BERTDataset(dev_df["text"], tokenizer, MAX_LEN, dev_df["labels"])

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

# # tokenize and encode sequences in the actual test set
# tokens = tokenizer.batch_encode_plus(train_df["text"].tolist(),
#                                          max_length = MAX_LEN,
#                                          pad_to_max_length=True,
#                                          truncation=True,
#                                          return_token_type_ids=True
#                                          )
# seq = torch.tensor(tokens['input_ids'])
# mask = torch.tensor(tokens['attention_mask'])
# token_type_ids = torch.tensor(tokens['token_type_ids'])
# targets = torch.tensor(train_df["labels"])
#
# train_dataset = TensorDataset(seq, mask, token_type_ids, targets)
#
# train_loader = DataLoader(train_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)

# 模型
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(pre_model)
        #         self.l2 = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, 6)

    def forward(self, ids, mask):
        # _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        _, features = self.roberta(ids, attention_mask=mask, return_dict=False)
        #         output_2 = self.l2(output_1)
        output = self.fc(features)
        return output


model = BERTClass()
model.to(device);

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = AdamW(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

def fin(x, threshold):
    if x >= threshold:
        return 1
    return 0


for epoch in range(EPOCHS):
    model.train()
    train_targets = []
    train_outputs = []
    for _, data in enumerate(tqdm(train_loader), 0):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        #         token_type_ids = data['token_type_ids'].to(device)
        targets = data['targets'].to(device)

        outputs = model(ids, mask)

        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
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
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            #             token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)
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


# torch.save(model.state_dict(), 'model1.bin')

test_dataset = BERTDataset(test["text"], tokenizer, MAX_LEN, is_dev = True)
test_loader = DataLoader(test_dataset, batch_size = TRAIN_BATCH_SIZE)


test_model = BERTClass()
test_model.to(device)
# test_model.load_state_dict(torch.load("../input/save-models/model.bin"))
test_model.eval()
preds = None
pred_end = None
preds = np.empty((len(test_dataset), 6))
pred_end = np.empty((len(test_dataset), 6))
n_batches = len(test_loader)
for step, batch in enumerate(tqdm(test_loader),0):
    ids = batch['ids'].to(device)
    mask = batch['mask'].to(device)
    # token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
    with torch.no_grad():
        outputs = test_model(ids, mask)
        pred_probs = torch.sigmoid(outputs)
    start_index = VALID_BATCH_SIZE * step
    end_index = (start_index + VALID_BATCH_SIZE if step != (n_batches - 1) else len(test_dataset))
    preds[start_index:end_index] = pred_probs.cpu().detach().numpy()

pred_end = [[fin(_,0.5) for _ in example] for example in preds]

sub = submit.copy()
sub['emotion'] = pred_end
sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))

print(sub.head())
# sub.to_csv('baseline.tsv', sep='\t', index=False)