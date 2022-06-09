import pandas as pd
import collections, json, random
def read_file(file_name):
    file_list = []
    with open(file_name,encoding="latin1") as f:
        for i,line in enumerate(f):
            if i==0:
                continue
            line = [item.strip().strip('"') for item in line.strip().split(";")]
            file_list.append(line)
    return file_list
rate_data = read_file("BX-Book-Ratings.csv")
book_attr = read_file("BX-Books.csv")
user_profile = read_file("BX-Users.csv")
# book profile
author_lable = {}
publisher_lable = {}
year_lable = {}
total_author_lable, total_publisher_lable, total_year_lable = 0, 0, 0
filter_book_profile = {}
book_profile = {}
for i in list(range(len(book_attr))):
    # author, year, publisher
    cur_author_lable = author_lable.setdefault(book_attr[i][2], total_author_lable)
    if cur_author_lable==total_author_lable:
        total_author_lable+=1
    cur_publisher_lable = publisher_lable.setdefault(book_attr[i][4], total_publisher_lable)
    if cur_publisher_lable==total_publisher_lable:
        total_publisher_lable+=1
    cur_year_lable = year_lable.setdefault(book_attr[i][3], total_year_lable)
    if cur_year_lable==total_year_lable:
        total_year_lable+=1
    filter_book_profile[book_attr[i][0]] = [cur_author_lable, cur_publisher_lable, cur_year_lable]
print(total_author_lable,total_publisher_lable,total_year_lable)
# user profile
user_profile_dict={}
filter_user_profile = {}
loc_lable = {}
total_loc_lable = 0
loc_dist = {}
age_dist = {}
num_no_age = 0
num_min = 0
num_max = 0
for i in list(range(len(user_profile))):
    user_profile_dict[user_profile[i][0]] = user_profile[i][1:]
    try:
        # age
        age = user_profile[i][-1]
        age = int(age)
        if age<40:
            num_min += 1
            age_lable = 0
        else:
            num_max += 1
            age_lable = 1
        age_dist[age] = age_dist.setdefault(age, 0) + 1
        # location
        loc = user_profile[i][1].split(",")[-1].strip()
        loc_dist[loc] = loc_dist.setdefault(loc, 0) + 1
        cur_lable = loc_lable.setdefault(loc, total_loc_lable)
        if cur_lable==total_loc_lable:
            total_loc_lable+=1
        filter_user_profile[user_profile[i][0]] = [cur_lable, age_lable]
    except:
        num_no_age+=1
print(total_loc_lable)
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
        book_id = rate_data[i][1]
        try:
            age = int(user_profile_dict[user_id][-1])
            user_profile = filter_user_profile[user_id]
            book_profile = filter_book_profile[book_id]
            filter_user_interaction_list[user_id].append(rate_data[i][1])
            filter_user_rate_list[user_id].append(int(rate_data[i][2]))
        except:
            continue
filter_user_interaction_follow, filter_user_rate_follow = {}, {}
filter_item_interaction_follow = collections.defaultdict(list)
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
loc_lable = {}
total_loc_lable = 0
for user in user_list:
    user_profile = filter_user_profile[id2user[user]]
    loc = user_profile[0]
    cur_loc_lable = loc_lable.setdefault(loc, total_loc_lable)
    if cur_loc_lable==total_loc_lable:
        total_loc_lable+=1
    filter_user_profile_follow[user] = [cur_loc_lable, user_profile[1]]
print(total_loc_lable)
author_lable = {}
publisher_lable = {}
year_lable = {}
v = 0
total_author_lable, total_publisher_lable, total_year_lable = 0, 0, 0
for book in item_list:
    book_attr = filter_book_profile[id2book[book]]
    cur_author_lable = author_lable.setdefault(book_attr[0], total_author_lable)
    if cur_author_lable==total_author_lable:
        total_author_lable+=1
    cur_publisher_lable = publisher_lable.setdefault(book_attr[1], total_publisher_lable)
    if cur_publisher_lable==total_publisher_lable:
        total_publisher_lable+=1
    cur_year_lable = year_lable.setdefault(book_attr[2], total_year_lable)
    if cur_year_lable==total_year_lable:
        total_year_lable+=1
    filter_book_profile_follow[book] = [cur_author_lable, cur_publisher_lable, cur_year_lable]
print(total_author_lable, total_publisher_lable, total_year_lable)
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




