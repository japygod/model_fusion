# -*- coding: utf-8 -*-

import csv
from shutil import copy
import re
import os
import torchvision


def processtext(text):
    text.replace('转发微博', '')
    text = re.sub('http(s)://[a-zA-Z0-9]+/[a-zA-Z0-9]+', '', text)
    return text


if __name__ == '__main__':
    model = torchvision.models.resnet50(pretrained=True)

    with open(r'F:\experiment\data.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        reader = list(reader)
        w = open(r'F:\experiment\data\data.csv', 'w', encoding='utf8')
        writer = csv.writer(w)
        row_length = len(reader[0])
        for i in range(len(reader)):
            if i < 1:
                writer.writerow(['text', ' piclist', 'label'])
                continue
            else:
                if len(reader[i][1]) <= 10:
                    continue
                else:
                    sentence = processtext(reader[i][1])



    # with open(r'F:\experiment\data\train.csv', 'r', encoding='utf8') as f:
    #     reader = csv.reader(f)
    #     reader = list(reader)
    #     row_length = len(reader[0])
    #     real_dest_dir = r'F:\experiment\real_img'
    #     false_dest_dir = r'F:\experiment\false_img'
    #     real_dir = r'F:\rumor_data\task3.z02等多个文件\task3\train\truth_pic'
    #     false_dir = r'F:\rumor_data\task3.z02等多个文件\task3\train\rumor_pic'
    #     for i in range(len(reader)):
    #         if i < 1:
    #             continue
    #         if reader[i][row_length-1] == '0':
    #             count = reader[i][2].count('.')
    #             if count == 1:
    #                 copy(real_dir + '\\' + reader[i][2], real_dest_dir + '\\' + str(i+1) + '.jpg')
    #             else:
    #                 ls = reader[i][2].split('\t')
    #                 for name in ls:
    #                     copy(real_dir + '\\' + name, real_dest_dir + '\\' + str(i+1)+'-'+name)
    #         else:
    #             count = reader[i][2].count('.')
    #             if count == 1:
    #                 copy(false_dir + '\\' + reader[i][2], false_dest_dir + '\\' + str(i+1) + '.jpg')
    #             else:
    #                 ls = reader[i][2].split('\t')
    #                 for name in ls:
    #                     copy(false_dir + '\\' + name, false_dest_dir + '\\' + str(i+1)+'-'+name)



