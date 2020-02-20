import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
from paper_config import configs
from paper_data_process import process_matching_data as train_paper
from test_data_process import process_matching_data as test_paper

from ad_news_data_process import process_news_data

from paper_model import bertEmbeddingLayer
from paper_model import matchingModel
from ad_model import adversarialDiscriminator
from ad_model import DiffLoss

# from domain_model_self import bertEmbeddingLayer
# from domain_model_self import localMatchingModule as matchingModel
# from domain_model_self import adversarialDiscriminator
# from domain_model_self import DiffLoss

import logging
import random
import json
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

torch.backends.cudnn.benchmark = True

# def train_matching_module(embedding_model, shared_encoder, matching_model, paper_data):
def generate_embedings(embedding_model, news_data, paper_data, device, mode):
    embedding_model.train()
    # embedding_model.eval()

    # generate_news_data
    # (input_ids, attention_masks, text_infos, 0)
    news_batch_input_ids = []
    news_batch_attention_masks = []
    news_batch_domain_labels = []
    for each_news in news_data:
        input_ids = each_news[0]
        attention_masks = each_news[1]
        # domain_label = 0
        # print(np.array(input_ids).shape)
        news_batch_input_ids.append(input_ids)
        news_batch_attention_masks.append(attention_masks)
        news_batch_domain_labels.append(0) 

    news_batch_input_ids = torch.tensor(news_batch_input_ids).to(device, non_blocking=True)
    news_batch_attention_masks = torch.tensor(news_batch_attention_masks).to(device, non_blocking=True)
    # print(news_batch_input_ids.size())
    _, _, news_batch_embeddings = embedding_model(news_batch_input_ids, news_batch_attention_masks)
    # news_batch_embeddings = news_batch_embeddings.to(domain_device)
    # news_batch_attention_masks = news_batch_attention_masks.to(domain_device)
    # news_batch_domain_labels = torch.tensor(news_batch_domain_labels).to(domain_device, non_blocking=True)
    # print(news_batch_embeddings.size())
    # exit()
    # print(paper_batch_embeddings.size())
    if(mode == "self"):
        return news_batch_embeddings
    # generate_paper_data
    #(input_ids_list_a, attention_masks_list_a), (input_ids_list_b, attention_masks_list_b), (per_input_ids_list_c, per_attention_masks_list_c) paper_batch_anchor_input_ids = []
    a_paper_batch_input_ids = []
    a_paper_batch_attention_masks = []

    b_paper_batch_input_ids = []
    b_paper_batch_attention_masks = []

    c_paper_batch_input_ids = []
    c_paper_batch_attention_masks = []


    for each_paper in paper_data:
        a_paper_inputs = each_paper[0][0]
        a_paper_attention_masks = each_paper[0][1]

        a_paper_batch_input_ids.extend(a_paper_inputs)
        a_paper_batch_attention_masks.extend(a_paper_attention_masks)

        b_paper_inputs = each_paper[1][0]
        b_paper_attention_masks = each_paper[1][1]

        b_paper_batch_input_ids.extend(b_paper_inputs)
        b_paper_batch_attention_masks.extend(b_paper_attention_masks)

        # c_paper_inputs = each_paper[2][0]
        # c_paper_attention_masks = each_paper[2][1]

        # c_paper_inputs_list = []
        # c_paper_attention_masks_list = []

        per_c_paper_inputs = each_paper[2][0]
        per_c_attention_masks = each_paper[2][1]

        for each_paper_inputs, each_attention_masks in zip(per_c_paper_inputs, per_c_attention_masks):
            # c_paper_inputs = torch.tensor(each_paper_inputs).to(device ,non_blocking=True)
            # c_paper_attention_masks = torch.tensor(each_attention_masks).to(device, non_blocking=True)
            

            c_paper_batch_input_ids.extend(each_paper_inputs)
            c_paper_batch_attention_masks.extend(each_attention_masks)





    a_paper_batch_input_ids = torch.tensor(a_paper_batch_input_ids).to(device, non_blocking=True)
    a_paper_batch_attention_masks = torch.tensor(a_paper_batch_attention_masks).to(device, non_blocking=True)

    # print(a_paper_batch_input_ids.size())

    b_paper_batch_input_ids = torch.tensor(b_paper_batch_input_ids).to(device, non_blocking=True)
    b_paper_batch_attention_masks = torch.tensor(b_paper_batch_attention_masks).to(device, non_blocking=True)
    # print(b_paper_batch_input_ids.size())


    c_paper_batch_input_ids = torch.tensor(c_paper_batch_input_ids).to(device, non_blocking=True)
    c_paper_batch_attention_masks = torch.tensor(c_paper_batch_attention_masks).to(device, non_blocking=True)
    # print(c_paper_batch_input_ids.size())
    total_paper_embeddings = []
    _, _, a_paper_batch_embeddings = embedding_model(a_paper_batch_input_ids, a_paper_batch_attention_masks)
    _, _, b_paper_batch_embeddings = embedding_model(b_paper_batch_input_ids, b_paper_batch_attention_masks)
    _, _, c_paper_batch_embeddings = embedding_model(c_paper_batch_input_ids, c_paper_batch_attention_masks)

    # print(c_paper_batch_embeddings.size())
    total_paper_embeddings.append(a_paper_batch_embeddings)
    total_paper_embeddings.append(b_paper_batch_embeddings)
    total_paper_embeddings.append(c_paper_batch_embeddings)

    total_paper_embeddings = torch.cat(total_paper_embeddings)
    # # total_paper_embeddings = roec
    # print(a_paper_batch_embeddings.size())
    # print(b_paper_batch_embeddings.size())
    # print(c_paper_batch_embeddings.size())
    # print(total_paper_embeddings.size())
    # exit()
    a_paper_batch_embeddings = a_paper_batch_embeddings.view(configs["domain_paper_batch_size"], configs["train_max_papers_each_author"], configs["hidden_size"])
    a_paper_batch_attention_masks = a_paper_batch_attention_masks.view(configs["domain_paper_batch_size"], configs["train_max_papers_each_author"], configs["train_max_paper_length"])
    # print(a_paper_batch_embeddings.size())


    b_paper_batch_embeddings = b_paper_batch_embeddings.view(configs["domain_paper_batch_size"], configs["train_max_papers_each_author"], configs["hidden_size"])
    b_paper_batch_attention_masks = b_paper_batch_attention_masks.view(configs["domain_paper_batch_size"], configs["train_max_papers_each_author"], configs["train_max_paper_length"])
    # print(a_paper_batch_embeddings.size())

    c_paper_batch_embeddings = c_paper_batch_embeddings.view(configs["domain_paper_batch_size"], configs["train_neg_sample"], configs["train_max_papers_each_author"], configs["hidden_size"])
    c_paper_batch_attention_masks = c_paper_batch_attention_masks.view(configs["domain_paper_batch_size"], configs["train_neg_sample"],configs["train_max_papers_each_author"], configs["train_max_paper_length"])
    # print(c_paper_batch_embeddings.size())

    return (news_batch_embeddings, a_paper_batch_embeddings, b_paper_batch_embeddings, c_paper_batch_embeddings, total_paper_embeddings)

    

def generate_data_batch(whole_data, batch_size):
	batch_data = []
	data_len = len(whole_data)
	for i in range(0, data_len, batch_size):
		batch_data.append(whole_data[i:min(i + batch_size, data_len)])
	return batch_data

def eval_hits(predictions, test_len):
    top_k = [1, 3, 5]
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    # print(predictions)
    # predictions = np.array(predictions).reshape((test_len, neg_sample + 1))
    # print(predictions)
    # print(predictions.shape)
    lengths = []
    gt = []
    for i in range(len(predictions)):
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(tmp_pre)
        # print(tmp_pre)
        true_index = np.where(rank == (len(tmp_pre) - 1))[0][0]
        if(true_index!=0):
            gt.append(0)
        else:
            gt.append(1)
        # true_index = np.where(rank == 0)[0][0]
        # if(len(rank) == 2):
            # print(rank)
            # print("total: {} true: {}".format(len(predictions[i]), true_index))
        lengths.append(len(rank))
        mrr += 1/(true_index +1)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr/test_len, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / test_len, 3)

    # print("hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    print(np.mean(lengths))
    return top_k, ratio_top_k, mrr, gt


def news_eval_hits(predictions, test_len, test_info):
    top_k = [1, 3, 5]
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    # print(predictions)
    # predictions = np.array(predictions).reshape((test_len, neg_sample + 1))
    # print(predictions)
    # print(predictions.shape)
    lengths = []
    gt = []
    for i in range(len(predictions)):
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(tmp_pre)
        # print(tmp_pre)
        # true_index = np.where(rank == (len(tmp_pre) - 1))[0][0]
        pre_ids = test_info[i][2][:-1]
        gt_ids = set(test_info[i][2][-1].split("---"))
        # if(pre_ids[rank[0]] in gt_ids):
        rerank_pre_ids = np.array(pre_ids)[rank]
        true_index = 100
        for gt in gt_ids:
            if(gt!=""):
                # print("re_rank: ",rerank_pre_ids)
                # print("gt_ids: ", gt_ids)
                try:
                    tmp_index = list(rerank_pre_ids).index(gt)
                # if(tmp_index == -1):
                #     print("error index")
                #     exit()
                except:
                    continue
                if(tmp_index<true_index):
                    true_index = tmp_index


        # if(true_index !=0):
            # print(test_info[i])
            # print(np.array(test_info[i][2])[rank])
            # print(tmp_pre[rank])
        # true_index = np.where(rank == 0)[0][0]
        # if(len(rank) == 2):
            # print(rank)
            # print("total: {} true: {}".format(len(predictions[i]), true_index))
        lengths.append(len(rank))
        mrr += 1/(true_index +1)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr/test_len, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / test_len, 3)

    # print("hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    print(np.mean(lengths))
    return top_k, ratio_top_k, mrr


def test_news_model(embedding_model, matching_model, test_news, test_info):
    embedding_model.eval()
    matching_model.eval()
    # embedding_model = bertEmbeddingLayer().to(bert_device)
    # matching_model = localMatchingModule().to(bert_device)
    test_loss = []
    matching_score = []
    with torch.no_grad():
        for ins_num in range(len(test_news)):
            instance_mean_loss = []
            tmp_matching_score = []
            instance = test_news[ins_num]
            a_paper_inputs, a_paper_attention_masks, \
            c_paper_inputs_list, c_paper_attention_masks_list = news_data2torch(instance, domain_device)
            # print(ins_num, a_paper_inputs.size())
            # print(a_paper_attention_masks.size())
            _, _, a_author_embeddings = embedding_model(a_paper_inputs, a_paper_attention_masks)

            # _, _, b_author_embeddings  = embedding_model(b_paper_inputs, b_paper_attention_masks)
            
            # a_author_embeddings = shared_encoder(a_author_embeddings)

            # b_author_embeddings = shared_encoder(b_author_embeddings)

            # pos_score = matching_model(a_author_embeddings, b_author_embeddings)
            # print(ins_num, pos_score, a_author_embeddings.size(), b_author_embeddings.size())
            # print("------neg---")
            for c_paper_inputs, c_paper_attention_masks in zip(c_paper_inputs_list, c_paper_attention_masks_list):
                _, _, c_author_embeddings  = embedding_model(c_paper_inputs, c_paper_attention_masks)
                # c_author_embeddings = shared_encoder(c_author_embeddings)
                neg_score = matching_model(a_author_embeddings, c_author_embeddings)
                # print(neg_score)
                # exit()
                # marginLoss = criterion(pos_score, neg_score, rank_y).unsqueeze(0)
                # instance_mean_loss.append(marginLoss)
                tmp_matching_score.append(neg_score.item())
            # tmp_matching_score.append(pos_score.item())
            matching_score.append(tmp_matching_score)

            # instance_mean_loss = torch.cat(instance_mean_loss)
            # instance_mean_loss = torch.mean(instance_mean_loss).unsqueeze(0)
            # test_loss.append(instance_mean_loss.item())
            # print(matching_score)
    top_k, ratio_top_k, mrr = news_eval_hits(matching_score, len(test_news), test_info)  
    print("news: hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    return ratio_top_k[0]

def news_data2torch(instance, device):
    # print(np.array(instance[0]).shape)
    a_paper_inputs = torch.tensor(instance[0]).to(device ,non_blocking=True)
    a_paper_attention_masks = torch.tensor(instance[1]).to(device, non_blocking=True)
    
    # b_paper_inputs = torch.tensor(instance[2]).to(device ,non_blocking=True)
    # b_paper_attention_masks = torch.tensor(instance[3]).to(device, non_blocking=True)
    
    c_paper_inputs_list = []
    c_paper_attention_masks_list = []

    per_c_paper_inputs = instance[2]
    per_c_attention_masks = instance[3]

    for each_paper_inputs, each_attention_masks in zip(per_c_paper_inputs, per_c_attention_masks):
        c_paper_inputs = torch.tensor(each_paper_inputs).to(device ,non_blocking=True)
        c_paper_attention_masks = torch.tensor(each_attention_masks).to(device, non_blocking=True)
        
        c_paper_inputs_list.append(c_paper_inputs)
        c_paper_attention_masks_list.append(c_paper_attention_masks)

    return a_paper_inputs, a_paper_attention_masks, c_paper_inputs_list, c_paper_attention_masks_list

def cluster_data2torch(instance, device):
    # print(np.array(instance[0]).shape)
    a_paper_inputs = torch.tensor(instance[0][0]).to(device ,non_blocking=True)
    a_paper_attention_masks = torch.tensor(instance[0][1]).to(device, non_blocking=True)
    
    b_paper_inputs = torch.tensor(instance[1][0]).to(device ,non_blocking=True)
    b_paper_attention_masks = torch.tensor(instance[1][1]).to(device, non_blocking=True)
    
    c_paper_inputs_list = []
    c_paper_attention_masks_list = []

    per_c_paper_inputs = instance[2][0]
    per_c_attention_masks = instance[2][1]

    for each_paper_inputs, each_attention_masks in zip(per_c_paper_inputs, per_c_attention_masks):
        c_paper_inputs = torch.tensor(each_paper_inputs).to(device ,non_blocking=True)
        c_paper_attention_masks = torch.tensor(each_attention_masks).to(device, non_blocking=True)
        
        c_paper_inputs_list.append(c_paper_inputs)
        c_paper_attention_masks_list.append(c_paper_attention_masks)

    return a_paper_inputs, a_paper_attention_masks, b_paper_inputs, b_paper_attention_masks, c_paper_inputs_list, c_paper_attention_masks_list


def test_cluster_model(embedding_model,matching_model, test_cluster):
    embedding_model.eval()
    matching_model.eval()
    test_loss = []
    matching_score = []
    with torch.no_grad():
        for test_ins_num in range(len(test_cluster)):
            instance_mean_loss = []
            tmp_matching_score = []
            instance = test_cluster[test_ins_num]
            a_paper_inputs, a_paper_attention_masks, \
            b_paper_inputs, b_paper_attention_masks, \
            c_paper_inputs_list, c_paper_attention_masks_list = cluster_data2torch(instance, domain_device)
            
            _, _, a_author_embeddings = embedding_model(a_paper_inputs, a_paper_attention_masks)

            _, _, b_author_embeddings  = embedding_model(b_paper_inputs, b_paper_attention_masks)
            # a_author_embeddings = torch.mean(a_author_embeddings, dim = 0).view(1, configs["hidden_size"])
            # b_author_embeddings = torch.mean(b_author_embeddings, dim = 0).view(1, configs["hidden_size"])

            # pos_loss, pos_score = contrastive_model(a_author_embeddings, b_author_embeddings, 1)
            # print(pos_loss)
            # instance_mean_loss.append(pos_loss.item())
            # pos_score = torch.cosine_similarity(a_author_embeddings, b_author_embeddings)
            pos_score = matching_model(a_author_embeddings, b_author_embeddings)
                        
            # print(test_ins_num, pos_score, a_author_embeddings.size(), b_author_embeddings.size())
            # print("-------neg--------")
            for c_paper_inputs, c_paper_attention_masks in zip(c_paper_inputs_list, c_paper_attention_masks_list):
                _, _, c_author_embeddings  = embedding_model(c_paper_inputs, c_paper_attention_masks)
                # c_author_embeddings = torch.mean(c_author_embeddings, dim = 0).view(1, configs["hidden_size"])
                # neg_loss, neg_score = contrastive_model(a_author_embeddings, c_author_embeddings, 0)
                # neg_score = torch.cosine_similarity(a_author_embeddings, c_author_embeddings)
                neg_score = matching_model(a_author_embeddings, c_author_embeddings)
                marginLoss = criterion(pos_score, neg_score, rank_y).unsqueeze(0)
                instance_mean_loss.append(marginLoss.item())
                tmp_matching_score.append(neg_score.item())
            tmp_matching_score.append(pos_score.item())
            # print(tmp_matching_score)
            # print(instance_mean_loss)
            matching_score.append(tmp_matching_score)
            # print(instance_mean_loss)
            # instance_mean_loss = torch.cat(instance_mean_loss)
            instance_mean_loss = np.mean(instance_mean_loss)
            
            test_loss.append(instance_mean_loss)
            # print(matching_score)
    top_k, ratio_top_k, mrr, _ = eval_hits(matching_score, len(test_cluster))

    print("paper: test_loss: {:.3f} hits@{} = {} mrr: {}".format(np.mean(test_loss), top_k, ratio_top_k, mrr))
    return ratio_top_k

if __name__ == "__main__":
    output_dir = "/home/chenbo/entity_linking/bert_generator/bert-base-multilingual-cased/"
    # output_dir = "/home/tangxiaobin/baseline-pretrain-bert-MLM/MLM-bert-model-epoch-0/"
    global bert_device
    global domain_device
    bert_device = torch.device("cuda:0")
    domain_device = torch.device("cuda:0")
    # device0 = torch.device("cuda:0")
    # bertModel = BertModel.from_pretrained(output_dir)
    bertTokenizer = BertTokenizer.from_pretrained(output_dir)

    news_data = process_news_data(bertTokenizer)
    # paper_data = process_matching_data(bertTokenizer)
    train_paper = train_paper(bertTokenizer)
    test_paper = test_paper(bertTokenizer)

    start = time.time()
    train_news = news_data.generate_train_news(configs["domain_news_batch_size"]*32, "TRAIN")
    random.shuffle(train_news)

    test_news, test_info = news_data.generate_test_news(3000)
    # print(len(test_news))
    # exit()
    # print()
    train_news_batches = generate_data_batch(train_news, configs["domain_news_batch_size"])

    
    print("News: total:{} batch: {} test: {} cost: {:3f}".format(len(train_news), len(train_news_batches), len(test_news), time.time() - start))
    # print("self news: ", len(self_news_batches))
    start = time.time()
    # train_papers = paper_data.generate_data(32, "TRAIN")
    train_papers = train_paper.generate_train_data(32)
    test_paper = test_paper.generate_test_data(100)
    # exit()

    random.shuffle(train_papers)
    train_paper_batches = generate_data_batch(train_papers, configs["domain_paper_batch_size"])

    print("Paper: total:{} batch: {} test: {} cost: {:3f}".format(len(train_papers), len(train_paper_batches), len(test_paper), time.time() - start))

    shared_embedding_model = bertEmbeddingLayer().to(domain_device)
    private_embedding_model = bertEmbeddingLayer().to(bert_device)
    matching_model = matchingModel(domain_device).to(domain_device)
    adversarial_model = adversarialDiscriminator().to(domain_device)

    # load_name = "../paper_level/paper_checkpoints/model_7_200"
    # load_name = "../paper_level/l2_model/l2_checkpoints/model_2_266"
    # load_name = "../paper_level/mlp_checkpoints/model_5_100"
    # load_name = "../paper_level/nomlp_checkpoints/model_5_100"


    # print("load")
    # # load_name = "./self_saved_model_via_mlp/model_5_3200"
    # load_name = "../matching_model/self_saved_model/model_9_3200"
    # checkpoint = torch.load(load_name, map_location = "cuda:1")
    # shared_embedding_model.load_state_dict(checkpoint['embebdding_model'])
    # matching_model.load_state_dict(checkpoint['local_matching_model'])
    # # shared_encoder.load_state_dict(checkpoint['shared_encoder'])
    # print("load_complete!")

    # shared_encoder = sharedModule().to(domain_device)
    # private_encoder =  targetPrivateModule().to(domain_device)

    # for p in embedding_model.parameters():
        # p.requires_grad = False

    criterion = nn.MarginRankingLoss(margin=1.0)
    rank_y = torch.tensor([-1.0], device = domain_device)
    # opt1= torch.optim.Adam([{'params': matching_model.parameters(), 'lr': configs["knrm_learning_rate"]},
    #                         {'params': embedding_model.parameters(), 'lr': configs["bert_learning_rate"]}])

    crossEntropyLoss = nn.CrossEntropyLoss()

    diffLoss = DiffLoss()


    # opt1 = torch.optim.Adam([{'params': shared_embedding_model.parameters(), 'lr': configs["bert_learning_rate"]},
    #                                 {'params': private_embedding_model.parameters(), 'lr': configs["bert_learning_rate"]}])
      
    # decayRate = 0.96
    # opt1_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = opt1, gamma=decayRate)

    optimizer = torch.optim.Adam([{'params': shared_embedding_model.parameters(), 'lr': configs["bert_learning_rate"]},
                                    {'params': private_embedding_model.parameters(), 'lr': configs["bert_learning_rate"]},
                                    {'params': adversarial_model.parameters(), 'lr': configs["adversarial_learning_rate"]},
                                    {'params': matching_model.parameters(), 'lr': configs["knrm_learning_rate"]}])
    # optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=decayRate)

    total_batch_num = len(train_paper_batches)
    assert total_batch_num == len(train_news_batches)
    # self_batch_num = len(self_news_batches)
    print("source only")
    load_name = "../paper_level/l2_model/l2_checkpoints/model_2_266"
    # load_name = "../paper_level/mlp_checkpoints/model_5_100"
    # load_name = "../paper_level/nomlp_checkpoints/model_5_100"
    print("load paper model: ", load_name)
    checkpoint = torch.load(load_name, map_location = "cuda:0")
    shared_embedding_model.load_state_dict(checkpoint['embedding_model'])
    matching_model.load_state_dict(checkpoint['matching_model'])
    # shared_encoder.load_state_dict(checkpoint['shared_encoder'])
    print("load_complete!")
    test_cluster_model(shared_embedding_model, matching_model, test_paper)
    test_news_model(shared_embedding_model, matching_model, test_news, test_info)

    load_name = "./fake_news_checkpoints/best_model_3_1600"
    print("load paper model: ", load_name)
    checkpoint = torch.load(load_name, map_location = "cuda:0")
    shared_embedding_model.load_state_dict(checkpoint['shared_embedding_model'])
    matching_model.load_state_dict(checkpoint['matching_model'])
    # shared_encoder.load_state_dict(checkpoint['shared_encoder'])
    print("load_complete!")
    test_cluster_model(shared_embedding_model, matching_model, test_paper)
    test_news_model(shared_embedding_model, matching_model, test_news, test_info)  
    # shared_embedding_model.train()
    # private_embedding_model.train()
    # matching_model.train()
    # adversarial_model.train()
    # # exit()
    # # train premier
    # # optimizer_scheduler.step()
    # max_hits = 0
    # min_test_loss = 10.0
    # # file_name = "./l2_3_adversarial_checkpoints/"
    # file_name = "./fake_news_checkpoints/"


