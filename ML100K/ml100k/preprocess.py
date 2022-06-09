import pandas as pd
import collections, json, random
rate_data = []
with open("ml-100k/u.data") as f:
    for i,line in enumerate(f):
        if i==0:
            continue
        line = [item for item in line.strip().split("\t")]
        line = line[:-1]
        rate_data.append(line)
movie_attr = []
with open("ml-100k/u.item", encoding = "ISO-8859-1") as f:
    for i,line in enumerate(f):
        if i==0:
            continue
        line = [item for item in line.strip().split("|")]
        
        line_per = []
        line_per.append(line[0])
        try:
            year = int(line[2].split("-")[-1])
            line_per.append(year)
        except:
            line_per.append(0)
        line_per.extend([int(sth) for sth in line[5:]])
        movie_attr.append(line_per)
user_profile = []
with open("ml-100k/u.user") as f:
    for i,line in enumerate(f):
        if i==0:
            continue
        line = [item for item in line.strip().split("|")]
        # age
        age = int(line[1])
        if age<40:
            age=0
        else:
            age=1
        line[1] = age
        # gender
        gender = line[2]
        if gender=="M":
            gender = 0
        else:
            gender = 1
        line[2] = gender
        user_profile.append(line)
# item profile
year_lable = {}
total_year_lable = 0
total_genre_lable = 19
filter_movie_profile = {}
movie_profile = {}
for i in list(range(len(movie_attr))):
    # year, genre
    if movie_attr[i][1]==0:
        continue
    cur_year_lable = year_lable.setdefault(movie_attr[i][1], total_year_lable)
    if cur_year_lable==total_year_lable:
        total_year_lable+=1
    filter_movie_profile[movie_attr[i][0]] = [cur_year_lable]
    filter_movie_profile[movie_attr[i][0]].extend(movie_attr[i][2:])
# user profile
user_profile_dict={}
filter_user_profile = {}
occ_lable, loc_lable = {}, {}
total_occ_lable, total_loc_lable = 0, 0
# occ_list, loc_dist = {}, {}
for i in list(range(len(user_profile))):
    try:
        # occupation
        occ = user_profile[i][3]
        cur_occ_lable = occ_lable.setdefault(occ, total_occ_lable)
        if cur_occ_lable==total_occ_lable:
            total_occ_lable+=1
        # location
        loc = user_profile[i][4]
        cur_loc_lable = loc_lable.setdefault(loc, total_loc_lable)
        if cur_loc_lable==total_loc_lable:
            total_loc_lable+=1
        filter_user_profile[user_profile[i][0]] = [user_profile[i][1], user_profile[i][2], cur_occ_lable, cur_loc_lable]
    except:
        pass
print(total_occ_lable, total_loc_lable)
print(total_year_lable, total_genre_lable)
# rate preprocess
num_0 = 0
user_list, item_list = set(), set()
filter_user_interaction_list = collections.defaultdict(list)
filter_user_rate_list = collections.defaultdict(list)
for i in list(range(len(rate_data))):
    rate = int(rate_data[i][2])
    if rate==0:
        num_0+=1
    else:
        user_id = rate_data[i][0]
        movie_id = rate_data[i][1]
        try:
            user_profile = filter_user_profile[user_id]
            movie_profile = filter_movie_profile[movie_id]
            filter_user_interaction_list[user_id].append(rate_data[i][1])
            filter_user_rate_list[user_id].append(int(rate_data[i][2]))
        except:
            continue
filter_user_interaction_follow, filter_user_rate_follow = {}, {}
filter_item_interaction_follow = collections.defaultdict(list)
# book -> item
user2id, book2id = {}, {}
id2user, id2book = {}, {}
total_user, total_book = 0, 0
for user, items in filter_user_interaction_list.items():
    # follow the MeLU setting
    if len(items)<13 or len(items)>100:
        continue
    user2id.setdefault(user, total_user)
    id2user[user2id[user]] = user
    if user2id[user]==total_user:
        total_user+=1
    items_reid = []
    user_reid = user2id[user]
    for item in items:
        book2id.setdefault(item, total_book)
        id2book[book2id[item]] = item
        item_reid = book2id[item]
        items_reid.append(item_reid)
        if item_reid==total_book:
            total_book+=1
        filter_item_interaction_follow[item_reid].append(user_reid)
    filter_user_interaction_follow[user_reid] = items_reid
    filter_user_rate_follow[user_reid] = filter_user_rate_list[user]
    
num_users = len(filter_user_interaction_follow)
print(num_users)
user_list = list(filter_user_interaction_follow.keys())
item_list = list(filter_item_interaction_follow.keys())
print(len(item_list))
filter_user_profile_follow, filter_book_profile_follow = {}, {}
occ_lable, loc_lable = {}, {}
total_occ_lable, total_loc_lable = 0, 0
for user in user_list:
    user_profile = filter_user_profile[id2user[user]]
    # occ
    occ = user_profile[2]
    cur_occ_lable = occ_lable.setdefault(occ, total_occ_lable)
    if cur_occ_lable==total_occ_lable:
        total_occ_lable+=1
    # loc
    loc = user_profile[3]
    cur_loc_lable = loc_lable.setdefault(loc, total_loc_lable)
    if cur_loc_lable==total_loc_lable:
        total_loc_lable+=1
    filter_user_profile_follow[user] = [user_profile[0], user_profile[1], cur_occ_lable, cur_loc_lable]
print(total_occ_lable, total_loc_lable)
year_lable = {}
total_year_lable = 0
total_genre_lable = 19
# book -> item
for book in item_list:
    book_attr = filter_movie_profile[id2book[book]]
    cur_year_lable = year_lable.setdefault(book_attr[0], total_year_lable)
    if cur_year_lable==total_year_lable:
        total_year_lable+=1
    filter_book_profile_follow[book] = [cur_year_lable]
    filter_book_profile_follow[book].extend(book_attr[1:])
print(total_year_lable, total_genre_lable)
random.seed(12345)
random.shuffle(user_list)
train_user_list = user_list[:int(0.7*num_users)]
valid_user_list = user_list[int(0.7*num_users):int(0.8*num_users)]
test_user_list = user_list[int(0.8*num_users):]
train_user_interaction, train_user_rate = {}, {}
for user in train_user_list:
    train_user_interaction[user] = filter_user_interaction_follow[user]
    train_user_rate[user] = filter_user_rate_follow[user]
valid_user_interaction, valid_user_rate = {}, {}
for user in valid_user_list:
    valid_user_interaction[user] = filter_user_interaction_follow[user]
    valid_user_rate[user] = filter_user_rate_follow[user]
test_user_interaction, test_user_rate = {}, {}
for user in test_user_list:
    test_user_interaction[user] = filter_user_interaction_follow[user]
    test_user_rate[user] = filter_user_rate_follow[user]

with open("user_warm_state.json", "w") as f:
    b = json.dumps(train_user_interaction)
    f.write(b)
with open("user_warm_state_y.json", "w") as f:
    b = json.dumps(train_user_rate)
    f.write(b)
with open("user_cold_state_valid.json", "w") as f:
    b = json.dumps(valid_user_interaction)
    f.write(b)
with open("user_cold_state_valid_y.json", "w") as f:
    b = json.dumps(valid_user_rate)
    f.write(b)
with open("user_cold_state_test.json", "w") as f:
    b = json.dumps(test_user_interaction)
    f.write(b)
with open("user_cold_state_test_y.json", "w") as f:
    b = json.dumps(test_user_rate)
    f.write(b)
with open("user_profile.json", "w") as f:
    b = json.dumps(filter_user_profile_follow)
    f.write(b)
with open("book_profile.json", "w") as f:
    b = json.dumps(filter_book_profile_follow)
    f.write(b)