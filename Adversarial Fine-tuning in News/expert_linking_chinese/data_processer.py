import json
import numpy as np
import random
from config import configs
import torch
from transformers import BertTokenizer
from collections import defaultdict
from operator import itemgetter
from harvesttext import HarvestText


class newsProcessor:
    def __init__(self, device):
        self.bertDir = "./bert_dir/bert-base-multilingual-cased/"
        self.bertTokenizer = BertTokenizer.from_pretrained(self.bertDir)
        self.ht = HarvestText()
    
    def is_chinese(self, uchar):
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
        else:
            return False
    


    def generate_test_news(self, mention2res, pro_news, name2aid2pid, pub_dict):
        # ins_dict = {}
        infos = []
        instance = []
        for news_info in mention2res:
            ori_name = news_info[0]
            can_name = news_info[1]
            pos_aid = news_info[2]
            news_id = news_info[-1]
            candidate_authors = list(set(name2aid2pid[can_name].keys()))
            each_ins = (ori_name, can_name, pos_aid, candidate_authors, news_id)
            tokenizer_ins, ins_info = self.tokenizer_emb(each_ins, pro_news, name2aid2pid, pub_dict)
            instance.append(tokenizer_ins)
            infos.append(ins_info)
           
        return instance, infos


    def tokenizer_emb(self, each, pro_news, name2aid2pid, pub_dict):
        ori_name, can_name, pos_aid, neg_author_lists, news_id = each
        key = ori_name + '-' + can_name + '-' + pos_aid + '-' + news_id

        filter_context, total_filter_context = self.preprocess_text(ori_name, pro_news[news_id])
        news_input_ids_list = []
        news_attention_masks_list = []
        for para in filter_context:
            context_token = self.bertTokenizer.encode_plus(para, max_length = configs["test_max_news_each_para_length"], truncation=True)

            input_ids = context_token["input_ids"]
            attention_masks = [1] * len(input_ids)
            padding_length = configs["test_max_news_each_para_length"] - len(input_ids)
            padding_input_ids = input_ids + [0] * padding_length
            # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
            padding_attention_masks = attention_masks + [0] * padding_length
            news_input_ids_list.append(padding_input_ids)
            news_attention_masks_list.append(padding_attention_masks)
        
        # neg_author
        per_neg_per_paper_input_ids = []
        per_neg_per_paper_attention_masks = []

        for neg_author_id in neg_author_lists:
            neg_per_paper_input_ids, neg_per_paper_attention_masks = self.get_author_encoder(neg_author_id, can_name, name2aid2pid, pub_dict)
            per_neg_per_paper_input_ids.append(neg_per_paper_input_ids)
            per_neg_per_paper_attention_masks.append(neg_per_paper_attention_masks)
        
        print_ids = neg_author_lists
        # print_ids.append(pos_aid)
        return (news_input_ids_list, news_attention_masks_list, per_neg_per_paper_input_ids, per_neg_per_paper_attention_masks), (key, print_ids)


    def preprocess_text(self, anchor_mention, news_info):
        news_content = news_info["content"]
        news_title = news_info["title"]
        sentence_list = self.ht.cut_sentences(news_content)
        merge_set = set()
        ne_set = set()
        filter_list = []
        filter_list.append((news_title, -1))
        ne_set.add(-1)
        for i, sent in enumerate(sentence_list):
            if(sent.find(anchor_mention) != -1):
                # merge_list.append()
                merge_set.add(i)
                # filter_list.append(sent)
                # merge_set.add()
        if(len(merge_set) == 0):
            return False, filter_list, []
        else:
            for i in merge_set:
                # filter_list.append(sentence_list[])
                filter_list.append((sentence_list[i], i))
                ne_set.add(i)
                for sent_s in range(i-6, i):
                    if (sent_s >= 0) and (sent_s not in ne_set):
                        filter_list.append((sentence_list[sent_s], sent_s))
                        ne_set.add(sent_s)

                # filter_list.append((sentence_list[i], i))
                # ne_set.add(i)

                # if (i + 3) < len(sentence_list):
                for sent_s in range(i, i+6):
                    if (sent_s < len(sentence_list)) and (sent_s not in ne_set):
                        filter_list.append((sentence_list[sent_s], sent_s))
                        ne_set.add(sent_s)
            
            sort_filter_sentence = sorted(filter_list, key = itemgetter(1))
            # print("mention: ", anchor_mention)
            # print("merge: ", merge_set)
            # print("sort: ", sort_filter_sentence)
            context_sentence = []
            for context in sort_filter_sentence:
                para = context[0]
                context_sentence.append(para)
            # print("context: ", context_sentence)
            # exit()
            total_context = " ".join(context_sentence)
            # seg_list = jieba.cut(total_context, cut_all=False)
            # seg_list = seg_list[:configs["max_news_length"]]
            # return True, total_context
            return context_sentence, total_context

    def get_author_encoder(self, author_id, author_name, name2aid2pid, pub_dict):
        input_ids_list = []
        attention_masks_list = []
        # print(author_name, author_id)
        # print(name2aid2pid)
        author_papers = name2aid2pid[author_name][author_id]
        # random.seed(configs["seed"])
        random.shuffle(author_papers)
        # sample_papers = random.sample(author_papers, configs["max_papers_each_author"])
        paper_count = 0
        for paper_id in author_papers:
            input_ids, attention_masks = self.paper_encoder(paper_id, pub_dict)
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_masks)
            paper_count += 1
            if(paper_count == configs["test_news_max_papers_each_author"]):
                break
        return input_ids_list, attention_masks_list

    def paper_encoder(self, paper_id, pub_dict):
        pid = paper_id.split('-')[0]
        papers_attr = pub_dict[pid]
        paper_str = self.get_res_abs(papers_attr)
        # print("paper:", paper_str)
        outputs = self.bertTokenizer.encode_plus(paper_str, max_length = configs["test_max_paper_length"],truncation=True)
        input_ids = outputs["input_ids"]
        # print(len(input_ids))
        attention_masks = [1] * len(input_ids)
        # type_ids = [0] * 

        padding_length = configs["test_max_paper_length"] - len(input_ids)
        padding_input_ids = input_ids + [0] * padding_length
        # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
        padding_attention_masks = attention_masks + [0] * padding_length
        # qa_padding_positions_id = qa_position_ids + [0] * padding_length  

        return padding_input_ids, padding_attention_masks


    def get_res_abs(self, papers_attr):
        # print(papers_attr)
        name_info = set()
        org_info = set()
        keywords_info = set()
        try:
            title = papers_attr["title"].strip().lower()
        except:
            title = ""

        try:
            abstract = papers_attr["abstract"].strip().lower()
        except:
            abstract = ""
        try:
            venue = papers_attr["venue"].strip().lower()
        except:
            venue = ""

        try:
            keywords = papers_attr["keywords"]
        except:
            keywords = []
        for ins in keywords:
            keywords_info.add(ins.strip().lower())

        for ins_author in papers_attr["authors"]:
            try:
                name = ins_author["name"].strip().lower()
            except:
                name = ""
            if(name != ""):
                name_info.add(name)
            
            try:
                orgnizations =ins_author["org"].strip().lower()
            except:
                orgnizations = ""
            if(orgnizations.strip().lower() != ""):
                org_info.add(orgnizations)

        name_str = " ".join(name_info).strip()
        org_str = " ".join(org_info).strip()
        keywords_str = " ".join(keywords_info).strip()


        # 论文信息，可用abstract替换keywords
        whole_info_str = title + ' ' + keywords_str + " " + name_str + " " + org_str + ' ' + venue + ' '

        return whole_info_str

