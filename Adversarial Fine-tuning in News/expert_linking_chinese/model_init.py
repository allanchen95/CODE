import json
from transformers import BertTokenizer, BertModel
from coad_model import bertEmbeddingLayer, matchingModel
import torch
import tqdm
import numpy as np

class expertLinking:
    def __init__(self, device):
        self.savedModel = "./saved_model/best_model_0_2400"
        # self.savedModel = "./saved_model/best_model_6_4800"
        self.device = device
        self.init_model()
    
    def init_model(self):
        self.bert_encoder = bertEmbeddingLayer().to(self.device)
        self.matching_model = matchingModel(self.device).to(self.device)

        print("Load model: ", self.savedModel)
        checkpoint = torch.load(self.savedModel, map_location = self.device)
        self.bert_encoder.load_state_dict(checkpoint['shared_embedding_model'])
        self.matching_model.load_state_dict(checkpoint['matching_model'])
        self.bert_encoder.eval()
        self.matching_model.eval()
        print("COAD model loads on device: {}".format(self.device))



    def perform_linking(self, candidates_tokens, author_info):
        matching_score = []
        with torch.no_grad():
            # predict for each news
            for ins_num in tqdm.tqdm(range(len(candidates_tokens))):
                tmp_matching_score = []
                instance = candidates_tokens[ins_num]

                # a for news infor, c for candidate authors from AMiner.
                a_paper_inputs, a_paper_attention_masks, \
                c_paper_inputs_list, c_paper_attention_masks_list = self.news_data2torch(instance)
                _, _, a_author_embeddings = self.bert_encoder(a_paper_inputs, a_paper_attention_masks)
                for c_paper_inputs, c_paper_attention_masks in zip(c_paper_inputs_list, c_paper_attention_masks_list):
                    _, _, c_author_embeddings  = self.bert_encoder(c_paper_inputs, c_paper_attention_masks)
                    # c_author_embeddings = shared_encoder(c_author_embeddings)
                    neg_score = self.matching_model(a_author_embeddings, c_author_embeddings)
                    tmp_matching_score.append(neg_score.item())
                # tmp_matching_score.append(pos_score.item())
                matching_score.append(tmp_matching_score)

        self.news_eval_hits(matching_score, len(candidates_tokens), author_info)  


    def news_data2torch(self, instance):
        # print(np.array(instance[0]).shape)
        a_paper_inputs = torch.tensor(instance[0]).to(self.device)
        a_paper_attention_masks = torch.tensor(instance[1]).to(self.device)
        
        c_paper_inputs_list = []
        c_paper_attention_masks_list = []

        per_c_paper_inputs = instance[2]
        per_c_attention_masks = instance[3]

        for each_paper_inputs, each_attention_masks in zip(per_c_paper_inputs, per_c_attention_masks):
            c_paper_inputs = torch.tensor(each_paper_inputs).to(self.device)
            c_paper_attention_masks = torch.tensor(each_attention_masks).to(self.device)
            
            c_paper_inputs_list.append(c_paper_inputs)
            c_paper_attention_masks_list.append(c_paper_attention_masks)

        return a_paper_inputs, a_paper_attention_masks, c_paper_inputs_list, c_paper_attention_masks_list



    def news_eval_hits(self, predictions, test_len, test_info):
        top_k = [1, 3, 5]
        mrr = 0
        top_k_metric = np.array([0 for k in top_k])

        lengths = []
        gt = []
        for i in range(len(predictions)):
            tmp_pre = np.array(predictions[i]) * -1
            rank = np.argsort(-tmp_pre)

            # test_info: [news_info, context, candidate_authors]
            pre_ids = test_info[i][1]
            news_info = test_info[i][0]

            rerank_pre_ids = np.array(pre_ids)[rank]
            rerank_scores = tmp_pre[rank]
            print("news_info: ", news_info)
            for author, score in zip(rerank_pre_ids, rerank_scores):
                print("Pre_author: {} Scores: {}".format(author, score))


