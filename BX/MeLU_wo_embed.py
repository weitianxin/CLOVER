import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from embeddings import item, user
import torch.nn as nn

class Linear(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear, self).forward(x)
        return out

class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc_user_dim = config.embedding_dim * 4
        self.fc_item_dim = config.embedding_dim * 4
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        self.adv = config.adv

        self.item_emb = item(config)
        self.user_emb = user(config)
        

        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, 1)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)
        # new for gender fair
        # self.fc_user = Linear(self.fc_user_dim, self.fc2_in_dim//2)
        # self.fc_item = Linear(self.fc_item_dim, self.fc2_in_dim//2)
        if self.adv:
            self.fc_user_1 = Linear(self.fc_user_dim, self.fc2_in_dim)
            self.fc_user_out = Linear(self.fc2_in_dim, config.num_gender)
            self.user_part = nn.Sequential(self.fc_user_1, nn.ReLU(), self.fc_user_out)
            self.local_part = nn.ModuleList([self.user_part, self.final_part])
            self.global_part = nn.ModuleList([self.item_emb, self.user_emb])
            self.adv_loss = nn.CrossEntropyLoss()
            # self.interaction = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)
            # self.final_part = nn.ModuleList([self.fc_user, self.fc_item, self.interaction])
            
    
    def forward(self, x, training = True):
        rate_idx = x[:, 0]
        genre_idx = x[:, 1:26]
        director_idx = x[:, 26:2212]
        actor_idx = x[:, 2212:10242]
        gender_idx = x[:, 10242]
        age_idx = x[:, 10243]
        occupation_idx = x[:, 10244]
        area_idx = x[:, 10245]

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        
        x = torch.cat((item_emb, user_emb), 1)
        x = self.final_part(x)
        # for fair
        # user_emb = F.relu(self.fc_user(user_emb))
        # item_emb = F.relu(self.fc_item(item_emb))
        # x = self.interaction(x)
        if self.adv:
            x_user = self.user_part(user_emb)
            adv_loss = self.adv_loss(x_user, gender_idx)
            return x, adv_loss
        return x



