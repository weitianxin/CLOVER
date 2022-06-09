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
        if config.remove_gender:
            self.fc_user_dim = config.embedding_dim * 3
        else:
            self.fc_user_dim = config.embedding_dim * 4
        self.fc_item_dim = config.embedding_dim * 4
        self.fc1_in_dim = self.fc_user_dim + self.fc_item_dim
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        self.adv = config.adv
        self.adv2 = config.adv2
        self.normalize = config.normalize
        self.all_adv = config.all_adv
        self.item_emb = item(config)
        self.user_emb = user(config)
        self.dual = config.dual
        # whether input item embedding
        self.item_adv = config.item_adv
        self.con = config.con
        self.bias = config.bias
        self.sim = config.sim
        self.out = config.out
        self.out2 = config.out2
        self.loss = config.loss
        self.fc1 = Linear(self.fc2_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        if config.loss==1:
            self.linear_out = Linear(self.fc2_out_dim, 5)
        else:
            self.linear_out = Linear(self.fc2_out_dim, 1)
        self.interaction = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)
        # new for gender fair
        self.fc_user = Linear(self.fc_user_dim, self.fc2_in_dim//2)
        self.fc_item = Linear(self.fc_item_dim, self.fc2_in_dim//2)
        self.final_part = nn.ModuleList([self.fc_user, self.fc_item, self.interaction])
        if self.adv:
            if config.out==0:
                self.fc_user_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
                # if config.item_adv:
                #     self.fc_item_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                #     self.fc_item_out = Linear(self.fc2_in_dim//2, config.num_gender)
                #     self.item_part = nn.Sequential(self.fc_item_1, nn.ReLU(), self.fc_item_out)
                #     self.local_part = nn.ModuleList([self.user_part, self.item_part, self.final_part])
                # else:
            elif config.out==1:
                if config.loss==0:
                    self.fc_user_1 = Linear(1, self.fc2_in_dim//2)
                    self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
                elif config.loss==1:
                    self.fc_user_1 = Linear(5, self.fc2_in_dim//2)
                    self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
            elif config.out==2:
                if config.loss==0:
                    self.fc_user_1 = Linear(2, self.fc2_in_dim//2)
                    self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
                    # self.fc_user_y_1 = Linear(1, self.fc2_in_dim//2)
                elif config.loss==1:
                    self.fc_user_1 = Linear(10, self.fc2_in_dim//2)
                    self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
                    # self.fc_user_y_1 = Linear(1, self.fc2_in_dim//2)
            elif config.out==3:
                if config.loss==0:
                    self.fc_user_1 = Linear(6, self.fc2_in_dim//2)
                    self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
                elif config.loss==1:
                    raise ValueError("only for loss=0")
            elif config.out==4:
                if config.loss==0:
                    self.fc_user_1 = Linear(1, self.fc2_in_dim//2)
                    self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
                elif config.loss==1:
                    raise ValueError("only for loss=0")
            elif config.out==5:
                self.fc_user_1 = Linear(self.fc2_in_dim//2+1, self.fc2_in_dim//2)
                self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
            elif config.out==6:
                self.fc_user_1 = Linear(self.fc2_in_dim//2+1, self.fc2_in_dim//2)
                self.fc_user_out = Linear(self.fc2_in_dim//2, 1)
            elif config.out==7:
                self.fc_user_1 = Linear(self.fc2_in_dim+1, self.fc2_in_dim//2)
                self.fc_user_out = Linear(self.fc2_in_dim//2, config.num_gender)
            elif config.out==8:
                self.fc_user_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                self.fc_user_out = Linear(self.fc2_in_dim//2, 1)
            elif config.out==9:
                self.fc_user_1 = Linear(self.fc2_in_dim+1, self.fc2_in_dim//2)
                self.fc_user_out = Linear(self.fc2_in_dim//2, 1)
            
            if self.item_adv:
                self.fc_item_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)

            if self.adv2:
                if config.out2==0:
                    self.fc_user_2 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                    self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                    # if config.item_adv:
                    #     self.fc_item_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                    #     self.fc_item_out = Linear(self.fc2_in_dim//2, config.num_gender)
                    #     self.item_part = nn.Sequential(self.fc_item_1, nn.ReLU(), self.fc_item_out)
                    #     self.local_part = nn.ModuleList([self.user_part, self.item_part, self.final_part])
                    # else:
                elif config.out2==1:
                    if config.loss==0:
                        self.fc_user_2 = Linear(1, self.fc2_in_dim//2)
                        self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                    elif config.loss==1:
                        self.fc_user_2 = Linear(5, self.fc2_in_dim//2)
                        self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                    # if self.item_adv:
                    #     self.fc_item_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                elif config.out2==2:
                    if config.loss==0:
                        self.fc_user_2 = Linear(2, self.fc2_in_dim//2)
                        self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                        # self.fc_user_y_1 = Linear(1, self.fc2_in_dim//2)
                    elif config.loss==1:
                        self.fc_user_2 = Linear(10, self.fc2_in_dim//2)
                        self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                        # self.fc_user_y_1 = Linear(1, self.fc2_in_dim//2)
                    # if self.item_adv:
                    #     self.fc_item_1 = Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
                elif config.out2==3:
                    if config.loss==0:
                        self.fc_user_2 = Linear(6, self.fc2_in_dim//2)
                        self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                    elif config.loss==1:
                        raise ValueError("only for loss=0")
                elif config.out2==4:
                    if config.loss==0:
                        self.fc_user_2 = Linear(1, self.fc2_in_dim//2)
                        self.fc_user_out_2 = Linear(self.fc2_in_dim//2, config.num_gender)
                    elif config.loss==1:
                        raise ValueError("only for loss=0")
            self.user_part = nn.Sequential(self.fc_user_1, nn.ReLU(), self.fc_user_out)
            # if config.out==0: # based on user embedding
            self.local_part = nn.ModuleList([self.user_part])
            if self.item_adv:
                self.local_part = nn.ModuleList([self.user_part, self.fc_item_1])
            if self.adv2:
                self.user_part_2 = nn.Sequential(self.fc_user_2, nn.ReLU(), self.fc_user_out_2)
                self.local_part.extend(self.user_part_2)
            # self.adv_global_part = nn.ModuleList([self.user_part])
            if config.inner_fc==1:
                self.local_max_part = nn.ModuleList([self.interaction])
            elif config.inner_fc==0:
                self.local_max_part = nn.ModuleList([self.fc_user, self.fc_item, self.interaction])# self.final_part
            if config.outer==0:
                if config.train_adv:
                    self.define_task_adv()
                    self.global_part = nn.ModuleList([self.item_emb, self.user_emb, self.fc_user, self.fc_item, self.interaction, self.task_adv])
                else:
                    self.global_part = nn.ModuleList([self.item_emb, self.user_emb, self.fc_user, self.fc_item, self.interaction])
            elif config.outer==1:
                if config.train_adv:
                    self.define_task_adv()
                    self.global_part = nn.ModuleList([self.item_emb, self.user_emb, self.fc_item, self.interaction, self.task_adv])
                else:
                    self.global_part = nn.ModuleList([self.item_emb, self.user_emb, self.fc_item, self.interaction])
                self.global_fix_part = nn.ModuleList([self.fc_user])
            # else:
            #     if config.out==2:
            #         # self.local_part = nn.ModuleList([self.fc_user_1, self.fc_user_y_1, self.fc_user_out])
            #         self.local_part = nn.ModuleList([self.user_part])
            #     elif config.out==1:
            #         self.local_part = nn.ModuleList([self.user_part])
            #     # else:
            #     #     raise ValueError("out only for 1 or 2")
            #     if config.inner_fc==1:
            #         self.local_max_part = nn.ModuleList([])#self.interaction
            #     elif config.inner_fc==0:
            #         self.local_max_part = nn.ModuleList([self.fc_user, self.fc_item, self.interaction])# self.final_part
            #     if config.train_adv:
            #         self.define_task_adv()
            #         self.global_part = nn.ModuleList([self.item_emb, self.user_emb, self.fc_user, self.fc_item, self.interaction, self.task_adv])
            #     else:
            #         self.global_part = nn.ModuleList([self.item_emb, self.user_emb, self.fc_user, self.fc_item, self.interaction])

            self.adv_loss = nn.CrossEntropyLoss()
            self.cos_sim = nn.CosineSimilarity()
            
    def define_task_adv(self):
        self.task_adv = nn.ParameterList()
        if self.dual:
            self.task_adv.append(nn.Parameter(torch.tensor([self.adv, self.adv], requires_grad=True)))
        else:
            self.task_adv.append(nn.Parameter(torch.tensor(self.adv, requires_grad=True)))
    
    def forward_user(self, x, counter=False):
        rate_idx = x[:, 0]
        genre_idx = x[:, 1:8]
        director_idx = x[:, 8:20]
        actor_idx = x[:, 20:24]
        gender_idx = x[:, 24]
        age_idx = x[:, 25]
        occupation_idx = x[:, 26]
        area_idx = x[:, 27]
        if counter:
            gender_idx = 1-gender_idx
        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        # for embed
        user_emb = F.relu(self.fc_user(user_emb))
        item_emb = F.relu(self.fc_item(item_emb))
        
        x_inter = torch.cat((item_emb, user_emb), 1)
        x = self.interaction(x_inter)
        return user_emb, item_emb, x, gender_idx

    def forward(self, x, y=None, x_all=None, y_all=None, training = True):
        con_user_emb, con_item_emb, con_x, con_gender_idx = self.forward_user(x, True)
        # con_user_emb = self.user_emb(con_gender_idx, age_idx, occupation_idx, area_idx)
        # con_user_emb = F.relu(self.fc_user(con_user_emb))
        # con_x = torch.cat((item_emb, con_user_emb), 1)
        # con_x = self.interaction(con_x)

        user_emb, item_emb, x, gender_idx = self.forward_user(x)

        # rate_idx = x[:, 0]
        # genre_idx = x[:, 1:26]
        # director_idx = x[:, 26:2212]
        # actor_idx = x[:, 2212:10242]
        # gender_idx = x[:, 10242]
        # age_idx = x[:, 10243]
        # occupation_idx = x[:, 10244]
        # area_idx = x[:, 28]

        # item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        # user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        # # for embed
        # user_emb = F.relu(self.fc_user(user_emb))
        # item_emb = F.relu(self.fc_item(item_emb))
        
        # x_inter = torch.cat((item_emb, user_emb), 1)
        # x = self.interaction(x_inter)
        output = [x]
        con_gender_idx = 1-gender_idx
        if self.adv:
            # ignore all_adv
            if self.all_adv:
                # all_adv_loss = 0
                # for x in x_all:
                x_user = torch.cat(x_all,0)
                user_emb, item_emb, x, gender_idx = self.forward_user(x_user)
                if self.normalize:
                    if self.loss==1:
                        x = F.softmax(x,dim=1)
                # user_emb, x, gender_idx = self.forward_user(x)
                y = torch.cat(y_all,0)
                if self.out==0:
                    x_user = self.user_part(user_emb)
                    adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==1:
                    x_in = x
                    x_user = self.user_part(x_in)
                    # x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==2:
                    if self.loss==0:
                        x_in = torch.cat([x, y],1)
                    elif self.loss==1:
                        y_one_hot = F.one_hot(torch.squeeze(y.long()-1), num_classes=5)
                        x_in = torch.cat([x, y_one_hot],1)
                    x_user = self.user_part(x_in)
                    # x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==3:
                    # change input 
                    if self.loss==0:
                        y_one_hot = F.one_hot(torch.squeeze(y.long()-1), num_classes=5)
                        x_in = torch.cat([x, y_one_hot],1)
                    else:
                        raise ValueError("only for loss=0")
                    x_user = self.user_part(x_in)
                    # x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==4:
                    # change input 
                    if self.loss==0:
                        y_minus = y-x
                        y_abs = torch.abs(y-x)
                        x_in = y_minus
                    else:
                        raise ValueError("only for loss=0")
                    x_user = self.user_part(x_in)
                    x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                # all_adv_loss+=adv_loss
                # adv_loss = all_adv_loss/len(x_all)
            else:
                if self.normalize:
                    if self.loss==1:
                        x = F.softmax(x,dim=1)
                if self.out==0:
                    # x_user = self.user_part(user_emb)
                    x_user = self.fc_user_out(F.relu(self.fc_user_1(user_emb)))
                    # con_gender_idx = 1-gender_idx
                    # con_user_emb = self.user_emb(con_gender_idx, age_idx, occupation_idx, area_idx)
                    # con_user_emb = F.relu(self.fc_user(con_user_emb))
                    # con_x_user = self.user_part(con_user_emb)
                    # gender_idx, x_user = torch.cat([gender_idx, con_gender_idx],0), torch.cat([x_user, con_x_user],0)
                    adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==1:
                    # x_in = x_inter
                    # x_in = torch.mean(x,0,True)
                    x_in = x
                    x_user = F.relu(self.fc_user_1(x_in))
                    if self.item_adv:
                        x_user += F.relu(self.fc_item_1(item_emb))
                    x_user = self.fc_user_out(x_user)
                    x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                elif self.out==2:
                    if self.loss==0:
                        x_in = torch.cat([x, y],1)
                        # x_in = torch.mean(torch.cat([x, y],1),0,True)
                    elif self.loss==1:
                        y_one_hot = F.one_hot(torch.squeeze(y.long()-1), num_classes=5)
                        x_in = torch.cat([x, y_one_hot],1)
                    # x_user = self.user_part(x_in)
                    x_user = F.relu(self.fc_user_1(x_in))
                    if self.item_adv:
                        x_user += F.relu(self.fc_item_1(item_emb))
                    x_user = self.fc_user_out(x_user)
                    # out_fc_user_1 = F.relu(self.fc_user_1(x_inter))
                    # out_fc_user_y_1 = F.relu(self.fc_user_y_1(y))
                    # x_user = self.fc_user_out(torch.cat([out_fc_user_1, out_fc_user_y_1], 1))
                    x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                elif self.out==3:
                    # change input 
                    if self.loss==0:
                        y_one_hot = F.one_hot(torch.squeeze(y.long()-1), num_classes=5)
                        x_in = torch.cat([x, y_one_hot],1)
                    else:
                        raise ValueError("only for loss=0")
                    x_user = self.user_part(x_in)
                    x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                elif self.out==4:
                    # change input 
                    if self.loss==0:
                        y_minus = y-x
                        y_abs = torch.abs(y-x)
                        # x_in = torch.cat([x, y, y_minus, y_abs],1)
                        x_in = y_minus
                        # x_in = torch.mean(x_in,0,True)
                        # x_in = torch.mean(torch.cat([x, y, y_minus, y_abs],1),0,True)
                    else:
                        raise ValueError("only for loss=0")
                    x_user = F.relu(self.fc_user_1(x_in))
                    if self.item_adv:
                        x_user += F.relu(self.fc_item_1(item_emb))
                    x_user = self.fc_user_out(x_user)
                    # x_user = torch.mean(x_user,0,True)
                    # adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                    adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==5:
                    # x_in = torch.cat([x, y, y_minus, y_abs],1)
                    x_in = torch.cat((user_emb, y),dim=1)
                    # x_in = torch.mean(x_in,0,True)
                    # x_in = torch.mean(torch.cat([x, y, y_minus, y_abs],1),0,True)
                    x_user = self.user_part(x_in)
                    x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                    # adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==6:
                    x_in = torch.cat((user_emb, y),dim=1)
                    x_in_con = torch.cat((con_user_emb, y),dim=1)
                    # x_in = user_emb
                    # x_in_con = con_user_emb
                    x_user = self.user_part(x_in)
                    x_user_con = self.user_part(x_in_con)
                    x_user = torch.mean(x_user,0)
                    x_user_con = torch.mean(x_user_con,0)
                    adv_loss = torch.abs(x_user-x_user_con)
                elif self.out==7:
                    # x_in = torch.cat([x, y, y_minus, y_abs],1)
                    x_in = torch.cat((user_emb, item_emb, y),dim=1)
                    # x_in = torch.mean(x_in,0,True)
                    # x_in = torch.mean(torch.cat([x, y, y_minus, y_abs],1),0,True)
                    x_user = self.user_part(x_in)
                    x_user = torch.mean(x_user,0,True)
                    adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                    # adv_loss = self.adv_loss(x_user, gender_idx)
                elif self.out==8:
                    # x_in = torch.cat((user_emb, y),dim=1)
                    # x_in_con = torch.cat((con_user_emb, y),dim=1)

                    x_in = user_emb
                    x_in_con = con_user_emb

                    x_user = self.user_part(x_in)
                    x_user_con = self.user_part(x_in_con)
                    x_user = torch.mean(x_user,0)
                    x_user_con = torch.mean(x_user_con,0)
                    adv_loss = torch.abs(x_user-x_user_con)
                elif self.out==9:
                    # x_in = torch.cat((user_emb, y),dim=1)
                    # x_in_con = torch.cat((con_user_emb, y),dim=1)

                    x_in = torch.cat((user_emb, item_emb, y),dim=1)
                    x_in_con = torch.cat((con_user_emb, item_emb, y),dim=1)
                    
                    x_user = self.user_part(x_in)
                    x_user_con = self.user_part(x_in_con)
                    x_user = torch.mean(x_user,0)
                    x_user_con = torch.mean(x_user_con,0)
                    adv_loss = torch.abs(x_user-x_user_con)
                
            if self.adv2:
                if self.out2==0:
                    # x_user = self.user_part(user_emb)
                    x_user = self.fc_user_out_2(F.relu(self.fc_user_2(user_emb)))
                    # con_gender_idx = 1-gender_idx
                    # con_user_emb = self.user_emb(con_gender_idx, age_idx, occupation_idx, area_idx)
                    # con_user_emb = F.relu(self.fc_user(con_user_emb))
                    # con_x_user = self.user_part(con_user_emb)
                    # gender_idx, x_user = torch.cat([gender_idx, con_gender_idx],0), torch.cat([x_user, con_x_user],0)
                    adv_loss += self.adv2/self.adv*self.adv_loss(x_user, gender_idx)
                elif self.out2==1:
                    # x_in = x_inter
                    # x_in = torch.mean(x,0,True)
                    x_in = x
                    x_user = F.relu(self.fc_user_2(x_in))
                    if self.item_adv:
                        x_user += F.relu(self.fc_item_1(item_emb))
                    x_user = self.fc_user_out_2(x_user)
                    x_user = torch.mean(x_user,0,True)
                    adv_loss += self.adv2/self.adv*self.adv_loss(x_user, gender_idx[0:1])
                elif self.out2==2:
                    if self.loss==0:
                        x_in = torch.cat([x, y],1)
                        # x_in = torch.mean(torch.cat([x, y],1),0,True)
                    elif self.loss==1:
                        y_one_hot = F.one_hot(torch.squeeze(y.long()-1), num_classes=5)
                        x_in = torch.cat([x, y_one_hot],1)
                    # x_user = self.user_part(x_in)
                    x_user = F.relu(self.fc_user_2(x_in))
                    if self.item_adv:
                        x_user += F.relu(self.fc_item_1(item_emb))
                    x_user = self.fc_user_out_2(x_user)
                    # out_fc_user_1 = F.relu(self.fc_user_1(x_inter))
                    # out_fc_user_y_1 = F.relu(self.fc_user_y_1(y))
                    # x_user = self.fc_user_out(torch.cat([out_fc_user_1, out_fc_user_y_1], 1))
                    x_user = torch.mean(x_user,0,True)
                    adv_loss += self.adv2/self.adv*self.adv_loss(x_user, gender_idx[0:1])
                elif self.out2==4:
                    # change input 
                    if self.loss==0:
                        y_minus = y-x
                        y_abs = torch.abs(y-x)
                        # x_in = torch.cat([x, y, y_minus, y_abs],1)
                        x_in = y_minus
                        # x_in = torch.mean(x_in,0,True)
                        # x_in = torch.mean(torch.cat([x, y, y_minus, y_abs],1),0,True)
                    else:
                        raise ValueError("only for loss=0")
                    x_user = F.relu(self.fc_user_2(x_in))
                    if self.item_adv:
                        x_user += F.relu(self.fc_item_1(item_emb))
                    x_user = self.fc_user_out_2(x_user)
                    x_user = torch.mean(x_user,0,True)
                    # adv_loss = self.adv_loss(x_user, gender_idx[0:1])
                    adv_loss += self.adv2/self.adv*self.adv_loss(x_user, gender_idx[0:1])
                
            output.append(adv_loss)
        # if self.con:
        # con_gender_idx = 1-gender_idx
        output.append(con_x)

        return output



