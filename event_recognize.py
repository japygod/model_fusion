
import os
from transformers import *
import pandas as pd
import json



# import numpy as np
# import random
# import torch
# import matplotlib.pylab as plt
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# from transformers import get_linear_schedule_with_warmup
#
# SEED = 123
# BATCH_SIZE = 16
# learning_rate = 2e-5
# weight_decay = 1e-2
# epsilon = 1e-8
#
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# def readFile(filename):
#     with open(filename, encoding='utf-8') as f:
#         content = f.readlines()
#         return content
#
# model_name = 'hfl/chinese-roberta-wwm-ext-large'
# # cache_dir = './sample_data/'
#
# tokenizer = BertTokenizer.from_pretrained(model_name)
#
# def convert_text_to_token(tokenizer, sentence, limit_size = 126):
#     tokens = tokenizer.encode(sentence[:limit_size])       # 直接截断
#     if len(tokens) < limit_size + 2:                       # 补齐（pad的索引号就是0）
#         tokens.extend([0] * (limit_size + 2 - len(tokens)))
#     return tokens
#
# input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]
#
# input_tokens = torch.tensor(input_ids)
# print(input_tokens.shape)
#
# def attention_masks(input_ids):
#     atten_masks = []
#     for seq in input_ids:                       # [10000, 128]
#         seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
#         atten_masks.append(seq_mask)
#     return atten_masks
#
# atten_masks = attention_masks(input_ids)
# attention_tokens = torch.tensor(atten_masks)
# print(attention_tokens.shape)
#
# train_data = TensorDataset(train_inputs, train_masks, train_labels)
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sample=train_sampler, batch_size=BATCH_SIZE)
#
# test_data = TensorDataset(test_inputs, test_masks, test_labels)
# test_sampler = RandomSampler(test_data)
# test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
#
# for i, (train, mask, label) in enumerate(train_dataloader):
#     # torch.Size([16, 128]) torch.Size([16, 128]) torch.Size([16, 1])
#     print(train.shape, mask.shape, label.shape)
#     break
#
# print('len(train_dataloader) = ', len(train_dataloader))    # 500
#
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2) # num_labels表示2个分类,好评和差评
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
#
# optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
#
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#      'weight_decay' : weight_decay
#     },
#     {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#      'weight_decay' : 0.0
#     }
# ]
#
# optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate, eps = epsilon)
#
# epochs = 2
# # training steps 的数量: [number of batches] x [number of epochs].
# total_steps = len(train_dataloader) * epochs
#
# # 设计 learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,
#                                             num_training_steps = total_steps)
#
# def binary_acc(preds, labels): # preds.shape = [16, 2] labels.shape = [16, 1]
#     # torch.max: [0]为最大值, [1]为最大值索引
#     correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
#     acc = correct.sum().item() / len(correct)
#     return acc
#
# import time
# import datetime
#
# def format_time(elapsed):
#     elapsed_rounded = int(round(elapsed))
#     return str(datetime.timedelta(seconds = elapsed_rounded))
#
# def train(model, optimizer):
#     t0 = time.time()
#     avg_loss, avg_acc = [],[]
#
#     model.train()
#     for step, batch in enumerate(train_dataloader):
#
#         # 每隔40个batch 输出一下所用时间.
#         if step % 40 == 0 and not step == 0:
#             elapsed = format_time(time.time() - t0)
#             print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
#
#         b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
#
#         output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#         loss, logits = output[0], output[1]      # loss: 损失, logits: predict
#
#         avg_loss.append(loss.item())
#
#         acc = binary_acc(logits, b_labels)       # (predict, label)
#         avg_acc.append(acc)
#
#         optimizer.zero_grad()
#         loss.backward()
#         clip_grad_norm_(model.parameters(), 1.0) # 大于1的梯度将其设为1.0, 以防梯度爆炸
#         optimizer.step()                         # 更新模型参数
#         scheduler.step()                         # 更新learning rate
#
#     avg_acc = np.array(avg_acc).mean()
#     avg_loss = np.array(avg_loss).mean()
#     return avg_loss, avg_acc
#
# def evaluate(model):
#     avg_acc = []
#     model.eval()         # 表示进入测试模式
#
#     with torch.no_grad():
#         for batch in test_dataloader:
#             b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
#
#             output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#
#             acc = binary_acc(output[0], b_labels)
#             avg_acc.append(acc)
#
#     avg_acc = np.array(avg_acc).mean()
#     return avg_acc
#
#
# for epoch in range(epochs):
#     train_loss, train_acc = train(model, optimizer)
#     print('epoch={},训练准确率={}，损失={}'.format(epoch, train_acc, train_loss))
#
#     test_acc = evaluate(model)
#     print("epoch={},测试准确率={}".format(epoch, test_acc))
#
# def predict(sen):
#
#     input_id = convert_text_to_token(tokenizer, sen)
#     input_token =  torch.tensor(input_id).long().to(device)            #torch.Size([128])
#
#     atten_mask = [float(i>0) for i in input_id]
#     attention_token = torch.tensor(atten_mask).long().to(device)       #torch.Size([128])
#
#     output = model(input_token.view(1, -1), token_type_ids=None, attention_mask=attention_token.view(1, -1))     #torch.Size([128])->torch.Size([1, 128])否则会报错
#     print(output[0])
#
#     return torch.max(output[0], dim=1)[1]

import numpy as np
import random
import torch
import matplotlib.pylab as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

class MyDataProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir,'train_data.txt')
        f = open(file_path,'r',encoding='utf-8')
        train_data = []
        index = 0
        for line in f.readlines():
            guid = "train-%d" % (index)
            line = line.replace('\n','').split('\t')
            text_a = tokenization.convert_to_unicode(str(line[0]))
            label = str(line[1])
            train_data.append(InputExample(guid=guid, text_a = text_a,text_b=None,label = label))
            index += 1
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir,'valid_data.txt')
        f = open(file_path,'r',encoding='utf-8')
        dev_data = []
        index = 0
        for line in f.readlines():
            guid = "train-%d" % (index)
            line = line.replace('\n','').split('\t')
            text_a = tokenization.convert_to_unicode(str(line[0]))
            label = str(line[1])
            dev_data.append(InputExample(guid=guid, text_a = text_a,text_b=None,label = label))
            index += 1
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir,'test_data.txt')
        f = open(file_path,'r',encoding='utf-8')
        test_data = []
        index = 0
        for line in f.readlines():
            guid = "train-%d" % (index)
            line = line.replace('\n','').split('\t')
            text_a = tokenization.convert_to_unicode(str(line[0]))
            label = str(line[1])
            test_data.append(InputExample(guid=guid, text_a = text_a,text_b=None,label = label))
            index += 1
        return test_data

    def get_labels(self):
        return ['0', '1']

from transformers import *


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
    # model = BertModel.from_pretrained('hfl/chinese-bert-wwm')
    # text = "今天天气是真的不错,阳光明媚，可以出去游玩"
    # print(tokenizer.tokenize(text))

    # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    # print(input_ids)
    # outputs = model(input_ids)
    # sequence_output = outputs[0]
    # pooled_output = outputs[1]
    # print(pooled_output)
    # print(sequence_output.shape)  ## 字向量
    # print(pooled_output)  ## 句向量，维数768
