import os
import torch
import torch.nn as nn
import json
from data_processer import newsProcessor
from model_init import expertLinking
import tqdm


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

# Load essential files


# Pre-align authors in news to the candidates in AMiner.
with open("news2aminer2aid.json", 'r') as files:
    name2aid2pid = json.load(files)
# Paper information
with open("aminer_pub_dict.json", 'r') as files:
    pub_dict = json.load(files)

# News information
with open("news_info.json", 'r') as files:
    process_news = json.load(files)

mention2res = []

for news_id, attr in process_news.items():
    mentions = attr["mention2result"]
    for author_name, attr in mentions.items():
        try:
            name_alias = attr["alias"]
        except:
            name_alias = ""
        # Ground truth to evaluate predictions. 
        aminer_id = "---".join(attr["ids"])
        tag = is_chinese(author_name)
        if(tag == False):
            mention2res.append((author_name, author_name, aminer_id, news_id))
        else:
            mention2res.append((author_name, name_alias, aminer_id, news_id))


# Device
local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_pro = newsProcessor(local_device)

# Process text info
test_news, test_info = data_pro.generate_test_news(mention2res, process_news, name2aid2pid, pub_dict)

# load model
linking_model = expertLinking(local_device)

# perform linking
linking_model.perform_linking(test_news, test_info)



