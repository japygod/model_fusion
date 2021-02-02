from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re

if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
    # text = "#海沧半马#12.10相约厦门我们不见不散🏼@厦门国际马拉松赛|厦门·厦门..."
    # print(tokenizer.tokenize(text))
    str1 = '#天津塘沽大爆炸#妈的，中国领导都是吃大便吗'
    t = re.findall("#[^#]+#", str1)
    print(t)
    str2 = '【云南西双版纳州景洪市发生4.9级地震震源深度12千米】@中国地震台网正式测定'

    # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    # outputs = model(input_ids)
    # sequence_output = outputs[0]
    # pooled_output = outputs[1]
    # print(sequence_output)
