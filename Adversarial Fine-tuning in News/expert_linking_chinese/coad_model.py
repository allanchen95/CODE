import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
import os
from torch.distributions import Categorical
from torch.autograd import Function
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import numpy as np
import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
bert_size = 768
hidden_size = 512


class bertEmbeddingLayer(nn.Module):
    def __init__(self):
        super(bertEmbeddingLayer, self).__init__()
        output_dir = "./bert_dir/bert-base-multilingual-cased/"
        self.bertModel = BertModel.from_pretrained(output_dir)
        self.Encoder = nn.Sequential(
            nn.Linear(bert_size, hidden_size),
            # nn.LeakyReLU(0.2, True)
            nn.Tanh()
        )       


    def forward(self, ins_token_inputs, ins_attention_mask, ins_position_ids = None):
        outputs = self.bertModel(input_ids = ins_token_inputs, attention_mask = ins_attention_mask)     
        # outputs:
            # 1. last_hidden_state: the last layer of the model. (batch_size, sequence_length, hidden_size)
            # 2. pooler_output: last layer hidden-state of the first token further processed by a Linear layer and a Tanh activation function.(batch_size, hidden_size)
            # 3. hidden_states: the output of each layer plus the initial embedding outputs. (batch_size, sequence_length, hidden_size)
        last_layer = outputs[0]
        pooler_out = outputs[1]
        # token_embeddings_last_4_layers = torch.cat((outputs[2][-1], outputs[2][-2], outputs[2][-3], outputs[2][-4]), 2) #[batch_size, seqence_len, 4 * 768]
        # cna
        output_encoder = self.Encoder(last_layer[:, 0])
        
        # sna
        # output_encoder = last_layer[:,0]
        # print(last_layer.size())
        # print(output_encoder.size())
        # exit()
        return pooler_out, ' ', output_encoder
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

    bin_size = 1.0 / (n_kernels - 1)  # score range from [-1, 1]
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
    # if n_kernels == 1:
        # return l_sigma

    l_sigma = [0.1] * (n_kernels - 1)
    l_sigma += [0.001]  # for exact match. small variance -> exact match
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
        self.mu = [0.9750, 0.9250, 0.8750, 0.8250, 0.7750, 0.7250, 0.6750, 0.6250,
        0.5750, 0.5250, 0.4750, 0.4250, 0.3750, 0.3250, 0.2750, 0.2250, 0.1750,
        0.1250, 0.0750, 0.0250, 0.000]
        self.sigma = [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
        0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
        0.1000, 0.1000, 0.1000 , 0.0010]
        self.mu = torch.FloatTensor(self.mu).to(device ,non_blocking=True)
        self.sigma = torch.FloatTensor(self.sigma).to(device ,non_blocking=True)

        # print(self.mu)
        # print(self.sigma)
        # exit()
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
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, 1),
            nn.Tanh()
            # nn.Dropout(0.5),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(128, 2)
        )
        self.relu_fnc = nn.Tanh()
        # self.learning2Rank = nn.Sequential(
        #     nn.Linear(self.n_bins, 1),
        #     nn.Tanh()
        # )

    def get_intersect_matrix(self, paper_embed, author_embed):

        # author_embed = author_embed.permute(1, 0)
        # sim_vec = paper_embed.mm(author_embed)
        paper_shape = paper_embed.size()
        author_shape = author_embed.size()
        # print(paper_shape)
        # print(author_shape)
        paper_embed = paper_embed.view(paper_shape[0], 1, 512)
        paper_embed = paper_embed.repeat(1, author_shape[0], 1).view(paper_shape[0]*author_shape[0], 512)
        author_embed = author_embed.repeat(paper_shape[0], 1)
        # print(paper_embed.size())
        # print(author_embed.size())
        sim_vec = self.relu_fnc(F.pairwise_distance(paper_embed, author_embed))
        # print(sim_vec)
        # print(sim_vec.size())
        # exit()
        sim_vec = sim_vec.view(paper_shape[0], author_shape[0])

        # print("11:",sim_vec.size())
        sim_vec = sim_vec.unsqueeze(-1)

        pooling_value = torch.exp((- ((sim_vec - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 1)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 0)
        return log_pooling_sum


    def forward(self, inputs_paper, inputs_author):
        # inputs_paper = torch.nn.functional.normalize(inputs_paper, p = 2, dim = 1)
        # inputs_author = torch.nn.functional.normalize(inputs_author, p = 2, dim = 1)
        log_pooling_sum = self.get_intersect_matrix(inputs_paper, inputs_author)
        # print(log_pooling_sum.size())
        # output = torch.squeeze(F.tanh(self.dense(log_pooling_sum)), 1)
        output = self.learning2Rank(log_pooling_sum)
        return output
