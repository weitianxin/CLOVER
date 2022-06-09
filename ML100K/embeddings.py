import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_year = config.num_year
        self.num_genre = config.num_genre
        self.embedding_dim = config.embedding_dim

        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year, 
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )
        
        # self.embedding_genre = torch.nn.Linear(
        #     in_features=self.num_genre,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )
        
        # self.embedding_director = torch.nn.Linear(
        #     in_features=self.num_director,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )
        
        # self.embedding_actor = torch.nn.Linear(
        #     in_features=self.num_actor,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )

    def forward(self, year_idx, genre_idx, vars=None):
        year_emb = self.embedding_year(year_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        return torch.cat((year_emb, genre_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.num_occupation = config.num_occupation
        self.num_zipcode = config.num_zipcode
        self.embedding_dim = config.embedding_dim
        self.remove_gender = config.remove_gender
        if not self.remove_gender:
            self.embedding_gender = torch.nn.Embedding(
                num_embeddings=self.num_gender,
                embedding_dim=self.embedding_dim
            )
        # self.embedding_gender = torch.nn.Embedding(
        #         num_embeddings=self.num_gender,
        #         embedding_dim=self.embedding_dim
        #     )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        if not self.remove_gender:
            gender_emb = self.embedding_gender(gender_idx)
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        if self.remove_gender:
            return torch.cat((age_emb, occupation_emb, area_emb), 1)
        else:
            return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
        # return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
