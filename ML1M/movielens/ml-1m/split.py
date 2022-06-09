import json
import numpy as np
with open("{}.json".format("warm_state_train"), encoding="utf-8") as f:
    dataset_split = json.loads(f.read())
with open("{}_y.json".format("warm_state_train"), encoding="utf-8") as f:
    dataset_split_y = json.loads(f.read())  
num_users_train = len(dataset_split.keys())
users_all = np.array(list(dataset_split.keys()))
arr = np.arange(num_users_train)
np.random.shuffle(arr)
num_valid = num_users_train//8
valid_users_index = arr[:num_valid]
train_users_index = arr[num_valid:]
valid_users = users_all[valid_users_index]
train_users = users_all[train_users_index]
dataset_split_train = {}
dataset_split_valid = {}
dataset_split_y_train = {}
dataset_split_y_valid = {}
for user in valid_users:
    dataset_split_valid[user] = dataset_split[user]
    dataset_split_y_valid[user] = dataset_split_y[user]
for user in train_users:
    dataset_split_train[user] = dataset_split[user]
    dataset_split_y_train[user] = dataset_split_y[user]
with open("{}.json".format("warm_state_train"), "w",encoding="utf-8") as f:
    b = json.dumps(dataset_split_train)
    f.write(b)
with open("{}_y.json".format("warm_state_train"), "w",encoding="utf-8") as f:
    b = json.dumps(dataset_split_y_train)
    f.write(b)

with open("{}.json".format("user_cold_state_valid"), "w", encoding="utf-8") as f:
    b = json.dumps(dataset_split_valid)
    f.write(b)

with open("{}_y.json".format("user_cold_state_valid"), "w",encoding="utf-8") as f:
    b = json.dumps(dataset_split_y_valid)
    f.write(b)
# print(dataset_split_valid)




