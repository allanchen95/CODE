import json
import numpy as np
import random
from ad_config import configs
# from mongo_online import mongo
# from bson import ObjectId
import torch
from transformers import BertTokenizer
from collections import defaultdict
from operator import itemgetter
from harvesttext import HarvestText

class process_news_data:
    def __init__(self, bertTokenizer):
        self.bertTokenizer = bertTokenizer
        self._load_raw_data()     
        self.ht = HarvestText()

    def is_chinese(self, uchar):
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
        else:
            return False

    def _load_raw_data(self):

        with open("../data/news_data/allNewsFilter.json", 'r') as files:
            self.news_data_0 = json.load(files)  

        with open("../data/news_data/430News.json", 'r') as files:
            self.news_data_1 = json.load(files)        
        
        self.news_data = {**self.news_data_0, **self.news_data_1}
        print("Train news_data: {} + {} = {}".format(len(self.news_data_0), len(self.news_data_1),len(self.news_data)))

        # data_dir = "/home/chenbo/entity_linking/data/news_data/"
        data_dir = "/data1/chenbo/"
        with open(data_dir + "news_name2aid2pid_mongo_2_9_whole.json", 'r') as files:
            self.name2aid2pid = json.load(files)

        with open(data_dir + "news_pub_dict_mongo_2_9_whole.json", 'r') as files:
            self.pub_dict = json.load(files)
        
        with open(data_dir + "news2aminer_whole_2_9.json", 'r') as files:
            self.process_news = json.load(files)

        # print("Test news_num: {}".format(len(self.process_news)))

        self.mention2res = []

        for news_id, attr in self.process_news.items():
            mentions = attr["mention2result"]
            for author_name, attr in mentions.items():
                try:
                    name_alias = attr["alias"]
                except:
                    name_alias = ""
                # a_id = attr["ids"]
                a_id = "---".join(attr["ids"])
                # print(author_name)
                tag = self.is_chinese(author_name)
                if(tag == False):
                    # print(author_name.isalpha())
                    self.mention2res.append((author_name, author_name, a_id, news_id))
                else:
                    # name_en = attr["alias"]
                    # a_id = attr["ids"]
                    # print(attr)
                    # print(name_alias)
                    self.mention2res.append((author_name, name_alias, a_id, news_id))

        print("Test news_num: {} Test_data: {}".format(len(self.process_news), len(self.mention2res)))


    def generate_test_news(self, ins_num):
        # ins_dict = {}
        infos = []
        instance = []
        test_list = self.mention2res
        # random.seed(configs["seed"])
        random.shuffle(test_list)
        count = 0
        err_c = 0
        for news_info in test_list:
            ori_name = news_info[0]
            can_name = news_info[1]
            pos_aid = news_info[2]
            news_id = news_info[-1]
            # print(news_info)
            # if(ori_name != "万小军"):
                # continue
            if(self.name2aid2pid.get(can_name) == None):
                continue
            candidate_authors = set(self.name2aid2pid[can_name].keys())
            # if(pos_aid not in candidate_authors):
                # print("error! --- ", ori_name, can_name)
                # exit()s
                # continue
            if((len(candidate_authors) - 1) < 1):
                # print("filter_can:", len(candidate_authors))
                continue
            # print("can:", len(candidate_authors))
            candidate_authors = list(candidate_authors)
            pos_aid_set = pos_aid.split("---")
            flag = False
            for ins_pos in pos_aid_set:
                if(ins_pos in candidate_authors):
                    flag = True
            if(flag == False):
                err_c += 1
                # print(err_c, news_info)
                continue
            # try:
            #     candidate_authors.remove(pos_aid)
            # except:
            #     print(news_info)
            #     # exit()
            #     continue

            # neg_author_lists = random.sample(candidate_authors, min(len(candidate_authors), 19))
            each_ins = (ori_name, can_name, pos_aid, candidate_authors, news_id)
            # print(each_ins)
            # exit()
            tag, tokenizer_ins, ins_info = self.test_tokenizer_padding(each_ins)
            # ins_dict[ins_info[0]] = ins_info[1]
            if(tag == False):
                continue
            count += 1
            instance.append(tokenizer_ins)
            infos.append(ins_info)
            if(count == ins_num):
                break
        # with open("test_news_info.json", 'w') as files:
            # json.dump(ins_dict, files, indent=4, ensure_ascii=False)
        return instance, infos

    def test_tokenizer_padding(self, each):
        ori_name, can_name, pos_aid, neg_author_lists, news_id = each
        key = ori_name + '-' + can_name + '-' + pos_aid + '-' + news_id

        flag, filter_context, total_filter_context = self.preprocess_text(ori_name, self.process_news[news_id]["content"])
        if(flag == False):
            # print("ffffff")
            # print(news_id, ori_name)
            return False, [],[]
        news_input_ids_list = []
        news_attention_masks_list = []
        for para in filter_context:
            context_token = self.bertTokenizer.encode_plus(para, max_length = configs["test_max_news_each_para_length"])

            input_ids = context_token["input_ids"]
            attention_masks = [1] * len(input_ids)
            # type_ids = [0] * 

            # news_input_ids = []
            # news_attention_masks = []

            padding_length = configs["test_max_news_each_para_length"] - len(input_ids)
            padding_input_ids = input_ids + [0] * padding_length
            # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
            padding_attention_masks = attention_masks + [0] * padding_length
            news_input_ids_list.append(padding_input_ids)
            news_attention_masks_list.append(padding_attention_masks)
        # context_token = self.bertTokenizer.encode_plus(total_filter_context, max_length = configs["max_news_length"])

        # input_ids = context_token["input_ids"]
        # attention_masks = [1] * len(input_ids)
        # padding_length = configs["max_news_length"] - len(input_ids)
        # padding_input_ids = input_ids + [0] * padding_length
        # # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
        # padding_attention_masks = attention_masks + [0] * padding_length

        # news_input_ids = []
        # news_attention_masks = []
        # news_input_ids.append(padding_input_ids)
        # news_attention_masks.append(padding_attention_masks)

        # pos_per_paper_input_ids, pos_per_paper_attention_masks = self.get_author_encoder(pos_aid, can_name)
        
        # neg_author
        per_neg_per_paper_input_ids = []
        per_neg_per_paper_attention_masks = []

        for neg_author_id in neg_author_lists:
            neg_per_paper_input_ids, neg_per_paper_attention_masks = self.get_author_encoder(neg_author_id, can_name)
            per_neg_per_paper_input_ids.append(neg_per_paper_input_ids)
            per_neg_per_paper_attention_masks.append(neg_per_paper_attention_masks)
        
        print_ids = neg_author_lists
        print_ids.append(pos_aid)
        return True, (news_input_ids_list, news_attention_masks_list, per_neg_per_paper_input_ids, per_neg_per_paper_attention_masks), (key, filter_context, print_ids)
        # return True, (news_input_ids, news_attention_masks, pos_per_paper_input_ids, pos_per_paper_attention_masks, per_neg_per_paper_input_ids, per_neg_per_paper_attention_masks)


    def preprocess_text(self, anchor_mention, news_content):
        sentence_list = self.ht.cut_sentences(news_content)
        merge_set = set()
        ne_set = set()
        filter_list = []
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
            return True, context_sentence, total_context

    def get_author_encoder(self, author_id, author_name):
        input_ids_list = []
        attention_masks_list = []

        author_papers = self.name2aid2pid[author_name][author_id]
        # random.seed(configs["seed"])
        random.shuffle(author_papers)
        # sample_papers = random.sample(author_papers, configs["max_papers_each_author"])
        paper_count = 0
        for paper_id in author_papers:
            tag, input_ids, attention_masks = self.paper_encoder(paper_id)
            if(tag == False):
                continue
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_masks)
            paper_count += 1
            if(paper_count == configs["test_news_max_papers_each_author"]):
                
                break
        return input_ids_list, attention_masks_list

    def paper_encoder(self, paper_id):
        pid = paper_id.split('-')[0]
        papers_attr = self.pub_dict[pid]
        tag, paper_str = self.get_res_abs(papers_attr)
        if(tag == False):
            return False, [], []
        # print("paper:", paper_str)
        outputs = self.bertTokenizer.encode_plus(paper_str, max_length = configs["test_max_paper_length"])
        input_ids = outputs["input_ids"]
        # print(len(input_ids))
        attention_masks = [1] * len(input_ids)
        # type_ids = [0] * 

        padding_length = configs["test_max_paper_length"] - len(input_ids)
        padding_input_ids = input_ids + [0] * padding_length
        # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
        padding_attention_masks = attention_masks + [0] * padding_length
        # qa_padding_positions_id = qa_position_ids + [0] * padding_length  

        return True, padding_input_ids, padding_attention_masks


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


        # whole_info = keywords_info
        # whole_info_str = title + ' ' + keywords_str + ' ' + name_str + " " + org_str + ' ' + venue
        whole_info_str = title + ' ' + keywords_str  + " " + org_str + ' ' + venue
        # print(whole_info_str)
        if(len(whole_info_str.strip().lower()) == 0):
            return False, ""
        else:
            return True, whole_info_str

    def generate_train_news(self, ins_num, mode):
        # instances_input_ids = []
        # instances_attention_masks = []
        instances = []
        count = 0
        break_tag = False
        # data_num = 0
        if(mode == "TRAIN"):
            news_data = self.news_data
        elif(mode == "TEST"):
            exit()
        else:
            print("ERROR! NO SUCH MODE!")
        news_id_list = list(news_data.keys())
        # random.seed(1)
        random.shuffle(news_id_list)
        for news_id in news_id_list:
            attr = news_data[news_id]
            entity_attr = attr["entity"]
            news_abs = attr["abstract"]
            news_content = attr["content"]            
            sentence_list = self.ht.cut_sentences(news_content)
            for i, sent in enumerate(sentence_list):
                # print(sent)
                each_ins = (sent, news_id)
                tag, each_input_ids, each_attention_masks = self.tokenizer_padding(each_ins)
                instances.append((each_input_ids, each_attention_masks))
                # instances_input_ids.append(each_input_ids)
                # instances_attention_masks.append(each_attention_masks)
                if(tag == False):
                    continue
                
                count += 1
                # instances.append(tokenizer_ins)
                if(count == ins_num):
                    break_tag = True
                    break
            if(break_tag == True):
                break

        return instances

    def tokenizer_padding(self, each):

        # text_infos = str(each[-1]) + "-" + str(each[0]) 
        tag, input_ids, attention_masks = self.tokenizer(each)
        if(tag == False):
            return False, [], [], []
        # total_data.append((input_ids, attention_masks, text_infos, 0))
        # exit()
        return True, input_ids, attention_masks


    def tokenizer(self, each):

        sent, news_id =  each
       
        # news_input_ids_list = []
        # news_attention_masks_list = []

        context_token = self.bertTokenizer.encode_plus(sent, max_length = configs["train_max_news_each_para_length"])

        input_ids = context_token["input_ids"]
        attention_masks = [1] * len(input_ids)
        # type_ids = [0] * 
        # print(sent)
        # print(len(input_ids))

        news_input_ids = []
        news_attention_masks = []

        padding_length = configs["train_max_news_each_para_length"] - len(input_ids)
        padding_input_ids = input_ids + [0] * padding_length
        # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
        padding_attention_masks = attention_masks + [0] * padding_length
        # news_input_ids_list.append(padding_input_ids)
        # news_attention_masks_list.append(padding_attention_masks)      
       
        # context_token = self.bertTokenizer.encode_plus(filter_context, max_length = configs["max_news_length"])

        # input_ids = context_token["input_ids"]
        # attention_masks = [1] * len(input_ids)
        # # type_ids = [0] * 

        # padding_length = configs["max_news_length"] - len(input_ids)
        # padding_input_ids = input_ids + [0] * padding_length
        # # qa_padding_token_type_ids = qa_token_type_ids + [1] * padding_length
        # padding_attention_masks = attention_masks + [0] * padding_length
        # # qa_padding_positions_id = qa_position_ids + [0] * padding_length  

        return True, padding_input_ids, padding_attention_masks
        # return True, news_input_ids_list, news_attention_masks_list



