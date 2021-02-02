import re
import jieba
import pandas as pd
import numpy as np
from numpy import *
from sklearn.model_selection import train_test_split

def pretext(content):  # 定义函数
    content1 = content.replace(' ', '')  # 去掉文本中的空格
    # print('\n【去除空格后的文本：】' + '\n' + content1)
    pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 只保留中英文、数字，去掉符号
    content2 = re.sub(pattern, '', content1)  # 把文本中匹配到的字符替换成空字符
    # print('\n【去除符号后的文本：】' + '\n' + content2)
    cutwords = jieba.lcut(content2, cut_all=False)  # 精确模式分词
    # print('\n【精确模式分词后:】' + '\n' + "/".join(cutwords))
    filepath2 = r'F:\tc毕业论文实验\all_stopwords.txt'
    stopwords = stopwordslist(filepath2)  # 这里加载停用词的路径
    words = ''
    for word in cutwords:  # for循环遍历分词后的每个词语
        if word not in stopwords:  # 判断分词后的词语是否在停用词表内
            if word != '\t':
                words += word
                words += "/"
    print('\n【去除停用词后的分词：】' + '\n' + words)
    return words
    # content3 = words.replace('/', '')  # 去掉文本中的斜线
    # lastword = pseg.lcut(content3)  # 使用for循环逐一获取划分后的词语进行词性标注
    # print('\n【对去除停用词后的分词进行词性标注：】' + '\n')
    # print([(words.word, words.flag) for words in lastword])  # 转换为列表

def stopwordslist(filepath2):    # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath2, 'r').readlines()]    #以行的形式读取停用词表，同时转换为列表
    return stopword


if __name__ == '__main__':
    incidence_eventdatadf = pd.read_excel(r'F:\tc毕业论文实验\ue_eventdata.xlsx')
    unincidence_eventdatadf = pd.read_excel(r'F:\tc毕业论文实验\e_eventdata.xlsx')
    incidence_eventdata = np.array(incidence_eventdatadf)
    unincidence_eventdata = np.array(unincidence_eventdatadf)
    data = []
    for li in incidence_eventdata:
        data.append([li[1], 1])
    for li in unincidence_eventdata:
        data.append([li[2], 0])
    data = array(data)
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size= 0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    with open('data\\train_data.txt', 'w', encoding='utf-8') as q:
        q.write('text_a\tlabel')
        t = ''
        for i in range(len(X_train)):
            temp = X_train[i].replace(' ', '')
            temp = temp.replace('\t', '')
            temp = temp.replace('\n', '')
            t = t + '\n' + str(temp)+'\t'+str(y_train[i])
            q.write(t)
            t = ''
    with open('data\\test_data.txt', 'w', encoding='utf-8') as q:
        q.write('text_a\tlabel')
        t = ''
        for i in range(len(X_test)):
            temp = X_test[i].replace(' ', '')
            temp = temp.replace('\t', '')
            temp = temp.replace('\n', '')
            t = t + '\n' + str(temp)+'\t'+str(y_test[i])
            q.write(t)
            t = ''
    with open('data\\valid_data.txt', 'w', encoding='utf-8') as q:
        q.write('text_a\tlabel')
        t = ''
        for i in range(len(X_valid)):
            temp = X_valid[i].replace(' ', '')
            temp = temp.replace('\t', '')
            temp = temp.replace('\n', '')
            t = t + '\n' + str(temp)+'\t'+str(y_valid[i])
            q.write(t)
            t = ''
    # t = ''
    # with open('pre_incidence_data.txt', 'w') as q:
    #     t = t + 'text_a\tlabel'
    #     q.write(t)
    #     t = ''
    #     for li in incidence_eventdata:
    #         t = t + str(pretext(li[1])) + '\t' + str(1)
    #         q.write(t)
    #         q.write('\n')
    #         t = ''
    # with open('pre_unincidence_data.txt', 'w') as q:
    #     for li in unincidence_eventdata:
    #         t = t + str(pretext(li[2])) + '\t' + str(0)
    #         q.write(t)
    #         q.write('\n')
    #         t = ''
