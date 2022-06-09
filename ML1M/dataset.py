import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import  re
import datetime
import pandas as pd
import json
from tqdm import tqdm

class movielens_1m(object):
    def __init__(self):
        self.user_data, self.item_data, self.score_data = self.load()

    def load(self):
        path = "movielens/ml-1m"
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'], 
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'], 
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data
    
def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = -1*torch.ones(1, 7).long()
    for i,genre in enumerate(str(row['genre']).split(", ")):
        idx = genre_list.index(genre)
        genre_idx[0, i] = idx
    director_idx = -1*torch.ones(1, 12).long()
    for i,director in enumerate(str(row['director']).split(", ")):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, i] = idx
    actor_idx = -1*torch.ones(1, 4).long()
    for i,actor in enumerate(str(row['actors']).split(", ")):
        idx = actor_list.index(actor)
        actor_idx[0, i] = idx
    # genre_idx = torch.zeros(1, 25).long()
    # for i,genre in enumerate(str(row['genre']).split(", ")):
    #     idx = genre_list.index(genre)
    #     genre_idx[0, idx] = 1
    # director_idx = torch.zeros(1, 8000).long()
    # for i,director in enumerate(str(row['director']).split(", ")):
    #     idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
    #     director_idx[0, idx] = 1
    # actor_idx = torch.zeros(1, 2000).long()
    # for i,actor in enumerate(str(row['actors']).split(", ")):
    #     idx = actor_list.index(actor)
    #     actor_idx[0, idx] = 1
    # max_genre = max(max_genre, len(str(row['genre']).split(", ")))
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1) #, len(str(row['genre']).split(", ")), len(str(row['director']).split(", ")), len(str(row['actors']).split(", "))


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list): 
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

class Metamovie(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metamovie, self).__init__()
        #self.dataset_path = args.data_root
        self.partition = partition
        self.adv = args.adv
        self.seed = args.seed
        #self.pretrain = pretrain
        
        self.dataset_path = args.data_root
        dataset_path = self.dataset_path
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

        self.dataset = movielens_1m()
        
        master_path = self.dataset_path
        if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
            self.movie_dict = {}
            max_genre, max_director, max_actor = 0, 0, 0
            for idx, row in self.dataset.item_data.iterrows():
                m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
                self.movie_dict[row['movie_id']] = m_info
            pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # key (num_id): 1 x num_item_feature_dim
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            self.user_dict = {}
            for idx, row in self.dataset.user_data.iterrows():
                u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
                self.user_dict[row['user_id']] = u_info
            pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
        else:
            self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))
        # key (num_id): 1  x num_user_feature_dim
        if partition == 'train':
            self.state = 'warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'warm_state'
                elif test_way == 'new_user_valid':
                    self.state = 'user_cold_state_valid'
                elif test_way == 'new_user_test':
                    self.state = 'user_cold_state_test'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            # str inside
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())            
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
        # reweight
        self.n_male_users, self.n_female_users = 0, 0
        if partition == 'train':
            for user_id in self.final_index:
                tmp_x = self.user_dict[int(user_id)]
                gender = tmp_x[0][0]
                if gender==0:
                    self.n_male_users+=1
                elif gender==1:
                    self.n_female_users+=1
            # print(self.n_male_users, self.n_female_users)

    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        # random.seed(53)
        if self.state=="warm_state":
            random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])
        
        support_x_app = None
        for m_id in tmp_x[indices[:-10]]:
            m_id = int(m_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
        query_x_app = None
        support_items = np.array([int(m_id) for m_id in tmp_x[indices[:-10]]])
        test_items = np.array([int(m_id) for m_id in tmp_x[indices[-10:]]])
        for m_id in tmp_x[indices[-10:]]:
            m_id = int(m_id)
            u_id = int(user_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
        # user_id and tmp_x is str
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1), user_id, test_items
        
    def __len__(self):
        return len(self.final_index)


class Metamovie_fair(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metamovie_fair, self).__init__()
        #self.dataset_path = args.data_root
        self.partition = partition
        #self.pretrain = pretrain
        
        self.dataset_path = args.data_root
        dataset_path = self.dataset_path
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

        self.dataset = movielens_1m()
        
        master_path = self.dataset_path
        if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
            self.movie_dict = {}
            for idx, row in self.dataset.item_data.iterrows():
                m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
                self.movie_dict[row['movie_id']] = m_info
            pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            self.user_dict = {}
            for idx, row in self.dataset.user_data.iterrows():
                u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
                self.user_dict[row['user_id']] = u_info
            pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
        else:
            self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))
        if partition == 'train' or partition == 'valid':
            self.state = 'warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'warm_state'
                elif test_way == 'new_user_test':
                    self.state = 'user_cold_state_test'
                elif test_way == 'new_user_valid':
                    self.state = 'user_cold_state_valid'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())            
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
        self.user_embedding = None
         

    def __getitem__(self, item):
        user_id = self.final_index[item]
        user_embedding = self.user_embedding[item]
        u_id = int(user_id)
        user_profile = self.user_dict[u_id]
        return user_profile.view(-1), torch.tensor(user_embedding).to(torch.float32)
        
    def __len__(self):
        return len(self.final_index)



# class Metamovie(Dataset):
#     def __init__(self, args, partition='train', test_way=None, path=None):
#         super(Metamovie, self).__init__()
#         #self.dataset_path = args.data_root
#         self.partition = partition
#         #self.pretrain = pretrain
        
#         self.dataset_path = args.data_root
#         dataset_path = self.dataset_path
#         rate_list = load_list("{}/m_rate.txt".format(dataset_path))
#         genre_list = load_list("{}/m_genre.txt".format(dataset_path))
#         actor_list = load_list("{}/m_actor.txt".format(dataset_path))
#         director_list = load_list("{}/m_director.txt".format(dataset_path))
#         gender_list = load_list("{}/m_gender.txt".format(dataset_path))
#         age_list = load_list("{}/m_age.txt".format(dataset_path))
#         occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
#         zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

#         self.dataset = movielens_1m()
        
#         master_path = self.dataset_path
#         if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
#             self.movie_dict = {}
#             for idx, row in self.dataset.item_data.iterrows():
#                 m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
#                 self.movie_dict[row['movie_id']] = m_info
#             pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
#         else:
#             self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
#         # hashmap for user profile
#         if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
#             self.user_dict = {}
#             for idx, row in self.dataset.user_data.iterrows():
#                 u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
#                 self.user_dict[row['user_id']] = u_info
#             pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
#         else:
#             self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))
#         if partition == 'train' or partition == 'valid':
#             self.state = 'warm_state'
#         else:
#             if test_way is not None:
#                 if test_way == 'old':
#                     self.state = 'warm_state'
#                 elif test_way == 'new_user':
#                     self.state = 'user_cold_state'
#                 elif test_way == 'new_item':
#                     self.state = 'item_cold_state'
#                 else:
#                     self.state = 'user_and_item_cold_state'
#         print(self.state)
#         with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
#             self.dataset_split = json.loads(f.read())
#         with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
#             self.dataset_split_y = json.loads(f.read())            
#         length = len(self.dataset_split.keys())
#         self.final_index = []
#         for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
#             u_id = int(user_id)
#             seen_movie_len = len(self.dataset_split[str(u_id)])

#             if seen_movie_len < 13 or seen_movie_len > 100:
#                 continue
#             else:
#                 self.final_index.append(user_id)
         

#     def __getitem__(self, item):
#         user_id = self.final_index[item]
#         u_id = int(user_id)
#         seen_movie_len = len(self.dataset_split[str(u_id)])
#         indices = list(range(seen_movie_len))
#         random.shuffle(indices)
#         tmp_x = np.array(self.dataset_split[str(u_id)])
#         tmp_y = np.array(self.dataset_split_y[str(u_id)])
        
#         support_x_app = None
#         for m_id in tmp_x[indices[:-10]]:
#             m_id = int(m_id)
#             tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
#             try:
#                 support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
#             except:
#                 support_x_app = tmp_x_converted
#         query_x_app = None
#         for m_id in tmp_x[indices[-10:]]:
#             m_id = int(m_id)
#             u_id = int(user_id)
#             tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
#             try:
#                 query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
#             except:
#                 query_x_app = tmp_x_converted
#         support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
#         query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
#         return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1)
        
#     def __len__(self):
#         return len(self.final_index)