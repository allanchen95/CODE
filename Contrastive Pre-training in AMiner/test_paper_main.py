import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
from paper_config import configs
# from paper_data_process import process_matching_data
from paper_data_process import process_matching_data as train_paper
from test_data_process import process_matching_data as test_paper
from paper_model import bertEmbeddingLayer
from paper_model import matchingModel
# from cons_model import ContrastiveLoss
# from domain_model import sharedModule
import logging
import random
import json
# logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

torch.backends.cudnn.benchmark = True
            

def eval_hits(predictions, test_len):
    top_k = [1, 3, 5]
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    # print(predictions)
    # predictions = np.array(predictions).reshape((test_len, neg_sample + 1))
    # print(predictions)
    # print(predictions.shape)
    lengths = []
    for i in range(len(predictions)):
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(tmp_pre)
        # print(len(tmp_pre))
        true_index = np.where(rank == (len(tmp_pre) - 1))[0][0]
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
    # print(np.mean(lengths))
    return top_k, ratio_top_k, mrr

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
            c_paper_inputs_list, c_paper_attention_masks_list = cluster_data2torch(instance, device)
            
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
            # exit()
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
    top_k, ratio_top_k, mrr = eval_hits(matching_score, len(test_cluster))

    print("cluster: test_loss: {:.3f} hits@{} = {} mrr: {}".format(np.mean(test_loss), top_k, ratio_top_k, mrr))
    return ratio_top_k

def generate_data_batch(whole_data, batch_size):
	batch_data = []
	data_len = len(whole_data)
	for i in range(0, data_len, batch_size):
		batch_data.append(whole_data[i:min(i + batch_size, data_len)])
	return batch_data

# def generate_embeddings(whole_data, embedding_model, batch_size):
    # embedding_model.train()
# def generate_embedings(embedding_model, batch_data):

#     for anchor, pos, neg_list in batch_data:
#         anchor_input = torch.tensor(anchor[0])
#         anchor_attention_mask = anchor[1]
    

if __name__ == "__main__":
    global var_dtype
    global device
    var_dtype = torch.half

    output_dir = "/home/chenbo/entity_linking/bert_generator/bert-base-multilingual-cased/"
    device = torch.device("cuda:0")
    bertTokenizer = BertTokenizer.from_pretrained(output_dir)

    # data_processor = process_matching_data(bertTokenizer)
    # start = time.time()
    # train_data = data_processor.generate_train_data(320)

    # test_data = data_processor.generate_test_data(30)
    # cost = time.time() - start
    train_data_processor = train_paper(bertTokenizer)
    test_data_processor = test_paper(bertTokenizer)
    start = time.time()
    train_data = train_data_processor.generate_train_data(64)

    test_data = test_data_processor.generate_test_data(1000)
    cost = time.time() - start
    # exit()
    embedding_model = bertEmbeddingLayer()
    matching_model = matchingModel(device)

    embedding_model.to(device)
    matching_model.to(device)

    # print("load")
    # load_name = "./self_saved_model_via_mlp/model_5_3200"
    load_name = "./l2_checkpoints/model_2_266"
    print("load model: {}".format(load_name))
    checkpoint = torch.load(load_name, map_location = "cuda:0")
    embedding_model.load_state_dict(checkpoint['embedding_model'])
    matching_model.load_state_dict(checkpoint['matching_model'])
    # shared_encoder.load_state_dict(checkpoint['shared_encoder'])
    print("load_complete!")

    batch_train_data = generate_data_batch(train_data, configs["local_accum_step"])
    total_batch_num = len(batch_train_data)
    print("#batch, train: {}|{} test: {} cost: {:.6f}".format(len(train_data), len(batch_train_data),len(test_data), cost))
    

    criterion = nn.MarginRankingLoss(margin=1.0)
    rank_y = torch.tensor([-1.0], device = device)

    optimizer = torch.optim.Adam([{'params': embedding_model.parameters(), 'lr': configs["train_bert_learning_rate"]},
                                {'params': matching_model.parameters(), 'lr': configs["train_knrm_learning_rate"]}])

    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=decayRate)

    file_name = "./l2_checkpoints/"
    max_hits = 0
    test_cluster_model(embedding_model, matching_model, test_data)