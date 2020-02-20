import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
import os
from paper_config import configs
from torch.distributions import Categorical
from torch.autograd import Function
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import numpy as np
import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class adversarialDiscriminator(nn.Module):
    def __init__(self):
        super(adversarialDiscriminator, self).__init__()
        # self.projection = nn.Sequential(
        #     nn.Linear(configs["hidden_size"], 384),
        #     nn.Dropout(0.5),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(384, 128),
        #     nn.Dropout(0.5),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(128, 2)
        # )
        self.projection = nn.Sequential(
            nn.Linear(configs["hidden_size"], 100),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 2)
        )

    # def forward(self, semantic_embeddings, alpha, tag):
    #     if(tag == "reverse"): 
    #         semantic_embeddings = ReverseLayerF.apply(semantic_embeddings, alpha)
    #     elif(tag == "normal"):
    #         semantic_embeddings = semantic_embeddings
    #     else:
    #         print("error model")
    #         exit()
    #     label_distirbution = self.projection(semantic_embeddings)
    #     # print(label_distirbution.size())
    #     softmax_scores = F.softmax(label_distirbution, dim = -1)
    #     # print(softmax_scores.size())
    #     # exit()
    #     return label_distirbution, softmax_scores

    def forward(self, semantic_embeddings, alpha, tag):
        semantic_embeddings = ReverseLayerF.apply(semantic_embeddings, alpha)
            # semantic_embeddings = semantic_embeddings
        label_distirbution = self.projection(semantic_embeddings)
        # print(label_distirbution.size())
        softmax_scores = F.softmax(label_distirbution, dim = -1)
        # print(softmax_scores.size())
        # exit()
        return label_distirbution, softmax_scores

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(configs["hidden_size"], 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs, outputs_mask):
        # (B, L) -> (B ,1)
        energy = self.projection(encoder_outputs)
        # (B)
        # weights = F.softmax(energy.squeeze(-1), dim = 0)
        weights = F.softmax(energy.squeeze(-1).masked_fill((1 - outputs_mask).byte(), float('-inf')), dim = 0)
        # (B) * (B) -> (B)
        # mask_weights = weights * outputs_mask
        # (B, L) * (B, 1) -> (L)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim = 0)

        return outputs, weights


class bertEmbeddingLayer(nn.Module):
    def __init__(self):
        super(bertEmbeddingLayer, self).__init__()
        output_dir = "/home/chenbo/entity_linking/bert_generator/bert-base-multilingual-cased/"
        self.bertModel = BertModel.from_pretrained(output_dir)
        self.Encoder = nn.Sequential(
            nn.Linear(configs["hidden_size"], configs["hidden_size"])
        )       


    def forward(self, ins_token_inputs, ins_attention_mask, ins_position_ids = None):
        outputs = self.bertModel(input_ids = ins_token_inputs, attention_mask = ins_attention_mask)     
        # outputs:
            # 1. last_hidden_state: the last layer of the model. (batch_size, sequence_length, hidden_size)
            # 2. pooler_output: last layer hidden-state of the first token further processed by a Linear layer and a Tanh activation function.(batch_size, hidden_size)
            # 3. hidden_states: the output of each layer plus the initial embedding outputs. (batch_size, sequence_length, hidden_size)
        last_layer = outputs[0]
        pooler_out = outputs[1]
        token_embeddings_last_4_layers = torch.cat((outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]), 2) #[batch_size, seqence_len, 4 * 768]
        # output_encoder = self.Encoder(last_layer[:, 0])
        return pooler_out, token_embeddings_last_4_layers[:, 0], last_layer[:,0]
        # return last_layer, pooler_out, token_embeddings_last_4_layers


def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class matchingModel(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, device):
        """
        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(matchingModel, self).__init__()
        self.n_bins = 21
        self.device = device
        # print(self.device)
        self.mu = torch.FloatTensor(kernal_mus(self.n_bins)).to(device ,non_blocking=True)
        self.sigma = torch.FloatTensor(kernel_sigmas(self.n_bins)).to(device ,non_blocking=True)

        self.mu = self.mu.view(1, 1, self.n_bins)
        self.sigma = self.sigma.view(1, 1, self.n_bins)
        # self.dense = nn.Linear(opt.n_bins, 1, 1)
        self.learning2Rank = nn.Sequential(
            # nn.Linear(self.n_bins, self.n_bins),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(self.n_bins, 1),
            # nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, 1),
            nn.Tanh()
            # nn.Dropout(0.5),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(128, 2)
        )

    def get_intersect_matrix(self, paper_embed, author_embed):

        author_embed = author_embed.permute(1, 0)
        sim_vec = paper_embed.mm(author_embed)
        # print("11:",sim_vec)
        sim_vec = sim_vec.unsqueeze(-1)

        pooling_value = torch.exp((- ((sim_vec - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 1)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 0)
        return log_pooling_sum


    def forward(self, inputs_paper, inputs_author):
        inputs_paper = torch.nn.functional.normalize(inputs_paper, p = 2, dim = 1)
        inputs_author = torch.nn.functional.normalize(inputs_author, p = 2, dim = 1)
        log_pooling_sum = self.get_intersect_matrix(inputs_paper, inputs_author)
        
        # print(log_pooling_sum.size())
        # output = torch.squeeze(F.tanh(self.dense(log_pooling_sum)), 1)
        output = self.learning2Rank(log_pooling_sum)
        # print(log_pooling_sum, output)
        return output


# class localMatchingModule(nn.Module):
#     def __init__(self):
#         super(localMatchingModule, self).__init__()
#         self.device = torch.device("cuda:1")
#         # self.bertencoder = bertEmbeddingLayer()
#         # self.dropout = nn.Dropout(0.5)  
#         self.n_bins = 11 
#         self.mus = torch.tensor([1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9], device = self.device)
#         self.sigmas = torch.tensor([1e-05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], device = self.device)
#         self.thres = torch.tensor(1e-10, device = self.device)
#         # self.norm = torch.tensor(configs["max_paper_length"], device = self.device)
#         self.norm = torch.tensor(0.01, device = self.device)

#         self.attentionLayer = nn.Sequential(
#             nn.Linear(self.n_bins, self.n_bins),
#             nn.Dropout(0.5),
#             # nn.Tanh()
#             nn.Linear(self.n_bins, 1)
#             # nn.Dropout(0.5),
#             # nn.LeakyReLU(0.2, True),
#             # nn.Linear(128, 2)
#         )
#         # r = tf.tanh(tf.nn.bias_add(tf.matmul(tf.reshape(inputs, [-1, size]), self.att_w), self.att_b))
#         self.learning2Rank = nn.Sequential(
#             nn.Linear(self.n_bins, self.n_bins),
#             nn.Dropout(0.5),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(self.n_bins, 1),
#             # nn.Dropout(0.5),
#             # nn.LeakyReLU(0.2, True),
#             # nn.Linear(self.n_bins, 1)
#             # nn.Tanh()
#             # nn.Dropout(0.5),
#             # nn.LeakyReLU(0.2, True),
#             # nn.Linear(128, 2)
#         )


    
#     def forward(self, a_author_embeddings, b_author_embeddings):
#         # paper_embeddings = paper_embeddings.squeeze(0)
#         # paper_masks = paper_masks.squeeze(0)
#         # paper_masks_ex = paper_masks.view(paper_masks.size()[0], 1)
#         # paper_masks_ex = paper_masks.unsqueeze(-1)
#         # author_masks_ex = author_masks.unsqueeze(-1)
#         # permute_author_masks = author_masks_ex.permute(0, 2, 1)

#         # masks = torch.einsum('ij, ajb -> aib', paper_masks_ex.float(), permute_author_masks.float())
#         # author_masks = author_masks.view(author_masks.size()[])
#         a_author_embeddings_norm = torch.nn.functional.normalize(a_author_embeddings, p = 2, dim = 1)
#         b_author_embeddings_norm = torch.nn.functional.normalize(b_author_embeddings, p = 2, dim = 1)

#         b_permute_author_embeddings = b_author_embeddings_norm.permute(1, 0)
#         sim_vec = a_author_embeddings_norm.mm(b_permute_author_embeddings)
#         # print("11:",sim_vec)
#         sim_vec_expand = sim_vec.unsqueeze(-1)
        
#         mus = self.mus.view(1, 1, self.n_bins)
#         sigma = self.sigmas.view(1, 1, self.n_bins)
#         # print("11:",sim_vec_expand)
#         sim_vec_expand_kernel = torch.exp(-(sim_vec_expand - mus).pow(2) / (2 * sigma.pow(2)))
#         # print("22: ", sim_vec_expand_kernel.size())
#         # exit()s
#         # sim_vec_expand_kernel = sim_vec_expand_kernel * masks.unsqueeze(-1)

#         sim_vec_kde = torch.sum(sim_vec_expand_kernel, dim = 1)

#         log_sim_vec_kde = torch.log(torch.max(sim_vec_kde, self.thres)) * self.norm

#         aggregated_sim_vec = torch.sum(log_sim_vec_kde, dim = 0)

#         matching_score = self.learning2Rank(aggregated_sim_vec)
        
#         return aggregated_sim_vec, matching_score