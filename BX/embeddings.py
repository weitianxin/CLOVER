import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_author = config.num_author
        self.num_year = config.num_year
        self.num_publisher = config.num_publisher
        self.embedding_dim = config.embedding_dim

        self.embedding_author = torch.nn.Embedding(
            num_embeddings=self.num_author, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_publisher =torch.nn.Embedding(
            num_embeddings=self.num_publisher,
            embedding_dim=self.embedding_dim
        )
        

    def forward(self, author_idx, publisher_idx, year_idx, vars=None):
        author_emb = self.embedding_author(author_idx)
        year_emb = self.embedding_year(year_idx)
        publisher_emb = self.embedding_publisher(publisher_idx)
        return torch.cat((author_emb, year_emb, publisher_emb), 1)


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_location = config.num_location
        self.num_age = config.num_age
        self.embedding_dim = config.embedding_dim
        self.remove_age = config.remove_age
        if not self.remove_age:
            self.embedding_age = torch.nn.Embedding(
                num_embeddings=self.num_age,
                embedding_dim=self.embedding_dim
            )
        # self.embedding_gender = torch.nn.Embedding(
        #         num_embeddings=self.num_gender,
        #         embedding_dim=self.embedding_dim
        #     )
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )

    def forward(self, location_idx, age_idx):
        if not self.remove_age:
            age_emb = self.embedding_age(age_idx)
        location_emb = self.embedding_location(location_idx)
        if self.remove_age:
            return location_emb
        else:
            return torch.cat((age_emb, location_emb), 1)
        # return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
