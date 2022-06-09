import copy
import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import util as utils
from dataset import Metamovie, Metamovie_fair
from logger import Logger
from MeLU import user_preference_estimator
from fair_cls import fair_classifier
import argparse
import torch
import time, heapq, itertools
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import collections
import logging
logging.getLogger().setLevel(logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser([],description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Clasification experiments.')

    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--cuda', type=int, default=1, help='cuda device')
    parser.add_argument('--task', type=str, default='multi', help='problem setting: sine or celeba')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32, help='number of tasks in each batch per meta-update')

    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimiser)')
    #parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--first_order', action='store_true', default=False, help='run first order approximation of CAVIA')

    parser.add_argument('--data_root', type=str, default="./movielens/ml-1m", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')
    parser.add_argument('--topk', type=int, default=3, help='num of workers to use')

    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')
    parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--second_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--num_epoch', type=int, default=60, help='num of workers to use')
    parser.add_argument('--num_genre', type=int, default=19, help='num of workers to use')
    parser.add_argument('--num_year', type=int, default=70, help='num of workers to use')
    # parser.add_argument('--num_director_unique', type=int, default=12, help='num of workers to use')
    # parser.add_argument('--num_genre_unique', type=int, default=7, help='num of workers to use')
    # parser.add_argument('--num_actor_unique', type=int, default=4, help='num of workers to use')

    parser.add_argument('--num_gender', type=int, default=2, help='num of workers to use')
    parser.add_argument('--num_age', type=int, default=2, help='num of workers to use')
    parser.add_argument('--num_occupation', type=int, default=21, help='num of workers to use')
    parser.add_argument('--num_zipcode', type=int, default=522, help='num of workers to use')
    parser.add_argument('--adv', type=float, default=0, help='adv coefficient')
    parser.add_argument('--adv2', type=float, default=0, help='adv2 coefficient')
    parser.add_argument('--fair', action='store_true', default=False, help='train fair classifier')
    parser.add_argument('--fair_lr', type=float, default=0.001, help='train fair classifier')
    parser.add_argument('--fm',action='store_true', default=False, help='get fairness metric')
    parser.add_argument('--remove_gender', action='store_true', default=False, help='train fair classifier')
    
    parser.add_argument('--nn', action='store_false', default=True,
                        help='whether add nn before embedding')
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')
    # testing arguments, the arguments: dual, train_adv, re_w, group, lam, con, sim, outer, bias, outer, inner_fc, more, schedule are just tested, and will not be actually used in the experiments
    parser.add_argument('--dual',action='store_true', default=False, help='dual trainable adv coefficient')
    parser.add_argument('--train_adv',action='store_true', default=False, help='trainable adv coefficient')
    parser.add_argument('--item_adv',action='store_true', default=False, help='item embedding input to adv, only 0|1')
    parser.add_argument('--re_w',action='store_true', default=False, help='reweight')
    parser.add_argument('--group',type=int, default=0, help='0: no group, 1: parity, 2: user perf, 3: value')
    parser.add_argument('--lam',type=float, default=0, help='0: no group, others: coeff')
    parser.add_argument('--con',action='store_true', default=False, help='counterfactual loss')
    parser.add_argument('--sim',action='store_true', default=False, help='sim loss')
    parser.add_argument('--bias',action='store_true', default=False, help='bias')
    parser.add_argument('--out',type=int, default=0, help='adversarial out type, 0: adv on user emb, 1: adv on logit, 2: adv on logit and label')
    parser.add_argument('--out2',type=int, default=-1, help='-1: no, else:有')
    parser.add_argument('--loss',type=int, default=0, help='0: mse loss, 1: ce loss')
    parser.add_argument('--outer',type=int, default=0, help='0: update fc outer, 1: not update fc outer')
    parser.add_argument('--inner_fc',type=int, default=0, help='0: update fc inner, 1: not update fc inner')
    parser.add_argument('--more',type=int, default=1, help='1: normal, 0: other')
    parser.add_argument('--adv_loss_power',type=float, default=1, help='1: normal, other: other')
    parser.add_argument('--disable_inner_max',action='store_true', default=False, help='1: x inner max, 0: remain inner max')
    parser.add_argument('--disable_inner_adv',action='store_true', default=False, help='1: x inner adv, 0: remain inner adv')
    parser.add_argument('--schedule',action='store_true', default=False, help='1: schedule, 0: not')
    parser.add_argument('--all_adv',action='store_true', default=False, help='1: schedule, 0: not')
    parser.add_argument('--normalize',action='store_true', default=False, help='1: normalize for adv input, 0: not')


    args = parser.parse_args()
    # use the GPU if available
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('Running on device: {}'.format(args.device))
    return args


def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)
    print('File saved in {}'.format(path))

    if os.path.exists(path + '.pkl') and not args.rerun:
        print('File has already existed. Try --rerun')
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)


    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    # initialise model
    model = user_preference_estimator(args).cuda()

    model.train()
    print(sum([param.nelement() for param in model.parameters()]))
    # set up meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
   # scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)

    # initialise logger
    logger = Logger()
    logger.args = args
    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    dataloader_train = DataLoader(Metamovie(args),
                                     batch_size=1,num_workers=args.num_workers)
    dataloader_valid = DataLoader(Metamovie(args, partition='test', test_way='new_user_valid'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
    dataloader_test = DataLoader(Metamovie(args, partition='test', test_way='new_user_test'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
    for epoch in range(args.num_epoch):
        
        x_spt, y_spt, x_qry, y_qry, test_items = [],[],[],[],[]
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            if len(x_spt)<args.tasks_per_metaupdate:
                x_spt.append(batch[0][0].cuda())
                y_spt.append(batch[1][0].cuda())
                x_qry.append(batch[2][0].cuda())
                y_qry.append(batch[3][0].cuda())
                test_items.append(batch[5].numpy()[0])
                if not len(x_spt)==args.tasks_per_metaupdate:
                    continue
            
            if len(x_spt) != args.tasks_per_metaupdate:
                continue

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)
            loss_pre = []
            loss_after = []
            group_pred = {0:[], 1:[]}
            group_perf = {0:[], 1:[]}
            group_item_pred = {0:collections.defaultdict(list), 1:collections.defaultdict(list)}
            # for re_w
            gender_list, gender_w = [], []
            for i in range(args.tasks_per_metaupdate): 
                gender = x_qry[i].detach().cpu().numpy()[0, 21]
                gender_list.append(gender)
            gender_list = np.array(gender_list)
            n_male_batch, n_female_batch = sum(gender_list==0), sum(gender_list==1)
            for sth in gender_list:
                if sth==0:
                    gender_w.append(1/(n_male_batch+n_female_batch*n_male_users/n_female_users))
                elif sth==1:
                    gender_w.append((n_male_users/n_female_users)/(n_male_batch+n_female_batch*n_male_users/n_female_users))
            for i in range(args.tasks_per_metaupdate): 
                gender = x_qry[i].detach().cpu().numpy()[0, 21]
                output = model(x_qry[i],y_qry[i], x_qry, y_qry)
                if args.loss==1:
                    loss_pre.append(F.cross_entropy(output[0], torch.squeeze(y_qry[i].long())-1).item())
                else:
                    loss_pre.append(F.mse_loss(output[0], y_qry[i]).item())
                fast_parameters = model.final_part.parameters()
                for weight in model.final_part.parameters():
                    weight.fast = None
                for k in range(args.num_grad_steps_inner):
                    output = model(x_spt[i],y_spt[i], x_spt, y_spt)
                    logits = output[0]
                    if args.loss==1:
                        loss = F.cross_entropy(logits,  torch.squeeze(y_spt[i].long())-1)
                    else:
                        loss = F.mse_loss(logits, y_spt[i])
                    # con not work
                    if args.con:
                        # now use coeff 1
                        loss += 1*F.mse_loss(output[1], logits)
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                    fast_parameters = []
                    for k, weight in enumerate(model.final_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]  
                        fast_parameters.append(weight.fast)         
                output = model(x_qry[i],y_qry[i], x_qry, y_qry)
                logits_q = output[0]

                # loss_q will be overwritten and just keep the loss_q on last update step.
                if args.loss==1:
                    loss_q = F.cross_entropy(logits_q,  torch.squeeze(y_qry[i].long())-1)
                else:
                    loss_q = F.mse_loss(logits_q, y_qry[i])
                loss_after.append(loss_q.item())
                if args.con:
                    # now use coeff 1
                    loss_q += 1*F.mse_loss(output[1], logits_q)
                task_grad_test = torch.autograd.grad(loss_q, model.parameters(), create_graph=True)
                
                for g in range(len(task_grad_test)):
                    if args.re_w:
                        meta_grad[g] += task_grad_test[g].detach()*gender_w[i]
                    else:
                        meta_grad[g] += task_grad_test[g].detach()/float(args.tasks_per_metaupdate)
                    
            # -------------- meta update --------------
            if args.lam:
                # parity loss
                if args.group==1:
                    group_loss = args.lam*torch.abs(torch.mean(torch.cat(group_pred[0],0))-torch.mean(torch.cat(group_pred[1],0))) 
                # value loss
                elif args.group==2:
                    group_loss = args.lam*torch.abs(torch.mean(torch.stack(group_perf[0]))-torch.mean(torch.stack(group_perf[1])))
                elif args.group==3:
                    male_pred, female_pred = group_item_pred[0], group_item_pred[1]
                    shared_items = list(set(male_pred.keys())&set(female_pred.keys()))
                    value_fairness = 0
                    for item in shared_items:
                        de = torch.mean(torch.cat(male_pred[item])) - torch.mean(torch.cat(female_pred[item]))
                        value_fairness += torch.abs(de)
                    group_loss = args.lam*value_fairness/len(shared_items)
                print(group_loss/args.lam)
                group_grad_test = torch.autograd.grad(group_loss, model.parameters())
                for g in range(len(group_grad_test)):
                        meta_grad[g] += group_grad_test[g].detach()
            meta_optimiser.zero_grad()
            # set gradients of parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c]
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            #scheduler.step()
            x_spt, y_spt, x_qry, y_qry, test_items = [],[],[],[],[]
            
            loss_pre = np.array(loss_pre)
            loss_after = np.array(loss_after)
            logger.train_loss.append(np.mean(loss_pre))
            logger.valid_loss.append(np.mean(loss_after))
            logger.train_conf.append(1.96*np.std(loss_pre, ddof=0)/np.sqrt(len(loss_pre)))
            logger.valid_conf.append(1.96*np.std(loss_after, ddof=0)/np.sqrt(len(loss_after)))
            logger.test_loss.append(0)
            logger.test_conf.append(0)
    
            # print current results
            logger.print_info(epoch, iter_counter, start_time)
            start_time = time.time()
            
            iter_counter += 1
        if epoch % (2) == 0:
            print('saving model at iter', epoch)
            logging.info('saving model at iter'+str(epoch))
            logger.valid_model.append(copy.deepcopy(model))
            utils.save_obj(logger, path)
            logging.info('Valid Eval')
            print('Valid Eval')
            evaluate_test(args, model, dataloader_valid)
            logging.info('Test Eval')
            print('Test Eval')
            evaluate_test(args, model, dataloader_test)

            

    return logger, model

def run_adv(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)
    print('File saved in {}'.format(path))

    if os.path.exists(path + '.pkl') and not args.rerun:
        print('File has already existed. Try --rerun')
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)


    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    # initialise model
    model = user_preference_estimator(args).cuda()

    model.train()
    print(sum([param.nelement() for param in model.parameters()]))
    # set up meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
   # scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)
    # initialise logger
    logger = Logger()
    logger.args = args
    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_local_init = [0 for _ in range(len(model.local_part.state_dict()))]
    meta_grad_global_init = [0 for _ in range(len(model.global_part.state_dict()))]
    if args.outer==1:
        meta_grad_global_fix_init = [0 for _ in range(len(model.global_fix_part.state_dict()))]
    dataloader_train = DataLoader(Metamovie(args),
                                     batch_size=1,num_workers=args.num_workers)
    dataloader_valid = DataLoader(Metamovie(args, partition='test', test_way='new_user_valid'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
    dataloader_test = DataLoader(Metamovie(args, partition='test', test_way='new_user_test'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
    num_batch = len(dataloader_train)//args.tasks_per_metaupdate
    T_max = args.num_epoch*num_batch
    for epoch in range(args.num_epoch):
        
        x_spt, y_spt, x_qry, y_qry = [],[],[],[]
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            if len(x_spt)<args.tasks_per_metaupdate:
                x_spt.append(batch[0][0].cuda())
                y_spt.append(batch[1][0].cuda())
                x_qry.append(batch[2][0].cuda())
                y_qry.append(batch[3][0].cuda())
                if not len(x_spt)==args.tasks_per_metaupdate:
                    continue
            
            if len(x_spt) != args.tasks_per_metaupdate:
                continue

            # initialise meta-gradient
            t = epoch*num_batch+step
            meta_grad_local = copy.deepcopy(meta_grad_local_init)
            meta_grad_global = copy.deepcopy(meta_grad_global_init)
            if args.outer==1:
                meta_grad_global_fix = copy.deepcopy(meta_grad_global_fix_init)
            group_pred = {0:[], 1:[]}
            # for re_w
            gender_list, gender_w = [], []
            for i in range(args.tasks_per_metaupdate): 
                gender = x_qry[i].detach().cpu().numpy()[0, 21]
                gender_list.append(gender)
            gender_list = np.array(gender_list)
            n_male_batch, n_female_batch = sum(gender_list==0), sum(gender_list==1)
            for sth in gender_list:
                if sth==0:
                    gender_w.append(1/(n_male_batch+n_female_batch*n_male_users/n_female_users))
                elif sth==1:
                    gender_w.append((n_male_users/n_female_users)/(n_male_batch+n_female_batch*n_male_users/n_female_users))
            loss_pre = []
            loss_after = []
            adv_loss_pre = []
            adv_loss_after = []
            if args.train_adv:
                adv_now = model.task_adv[0]
            else:
                adv_now = args.adv
            for i in range(args.tasks_per_metaupdate): 
                gender = x_qry[i].detach().cpu().numpy()[0, 21]
                if args.re_w:
                    if args.dual:
                        if gender==0:
                            adv = adv_now[0]
                        elif gender==1:
                            adv = adv_now[1]
                    else:
                        adv = adv_now
                else:
                    if args.dual:
                        if gender==0:
                            adv = adv_now[0]
                        elif gender==1:
                            adv = adv_now[1]
                    else:
                        adv = adv_now
                        if args.schedule:
                            adv = 2/(1+np.exp(-10*t/T_max))-1
                item_adv = args.item_adv
                output = model(x_qry[i],y_qry[i], x_qry, y_qry)
                logits, adv_loss = output[0], output[1]
                adv_loss_pre.append(adv_loss.item())
                if args.loss==1:
                    loss_pre.append(F.cross_entropy(logits, torch.squeeze(y_qry[i].long())-1).item())
                else:
                    loss_pre.append(F.mse_loss(logits, y_qry[i]).item())
                fast_parameters = model.local_part.parameters()
                for weight in model.local_part.parameters():
                    weight.fast = None
                # local max update
                fast_parameters_max = model.local_max_part.parameters()
                for weight in model.local_max_part.parameters():
                    weight.fast = None
                for l in range(args.num_grad_steps_inner):
                    output = model(x_spt[i], y_spt[i], x_spt, y_spt)
                    logits, adv_loss = output[0], output[1]
                    # local max update
                    if args.loss==1:
                        if args.disable_inner_max:
                            loss_max = F.cross_entropy(logits, torch.squeeze(y_spt[i].long())-1)
                        else:
                            loss_max = F.cross_entropy(logits, torch.squeeze(y_spt[i].long())-1) - adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                    else:
                        if args.disable_inner_max:
                            loss_max = F.mse_loss(logits, y_spt[i])
                        else:
                            loss_max = F.mse_loss(logits, y_spt[i]) - adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                    if args.con:
                        # now use coeff 1
                        loss_max += 1*F.mse_loss(output[2], logits)
                    
                    grad_max = torch.autograd.grad(loss_max, fast_parameters_max, create_graph=True)   
                    fast_parameters_max = []
                    for k, weight in enumerate(model.local_max_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad_max[k] #create weight.fast 
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad_max[k]
                        fast_parameters_max.append(weight.fast)
                    if not args.disable_inner_adv:
                        if args.loss==1:
                            # loss =  F.cross_entropy(logits, torch.squeeze(y_spt[i].long())-1) + adv*adv_loss
                            loss = adv_loss**args.adv_loss_power/args.adv_loss_power
                        else:
                            # loss =  F.mse_loss(logits, y_spt[i]) + adv*adv_loss
                            loss = adv_loss**args.adv_loss_power/args.adv_loss_power
                        if args.con:
                            # now use coeff 1
                            loss += 1*F.mse_loss(output[2], logits)
                        # print(torch.autograd.grad(adv*adv_loss, fast_parameters, create_graph=True, allow_unused=True))
                        # print(torch.autograd.grad(F.mse_loss(logits, y_spt[i]) + adv*adv_loss, fast_parameters, create_graph=True, allow_unused=True))
                        grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                        # print(grad)
                        
                        fast_parameters = []
                        for k, weight in enumerate(model.local_part.parameters()):
                            if weight.fast is None:
                                weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                            else:
                                weight.fast = weight.fast - args.lr_inner * grad[k]  
                            fast_parameters.append(weight.fast)     
                    
                output = model(x_qry[i], y_qry[i], x_qry, y_qry)
                logits_q, adv_loss = output[0], output[1]
                adv_loss_after.append(adv_loss.item())
                group_pred[gender].append(logits_q)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                if args.loss==1:
                    loss_q_global = F.cross_entropy(logits_q, torch.squeeze(y_qry[i].long())-1) - adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                else:
                    loss_q_global = F.mse_loss(logits_q, y_qry[i]) - adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                if args.con:
                    # now use coeff 1
                    loss_q_global += 1*F.mse_loss(output[2], logits_q)
                task_grad_test_global = torch.autograd.grad(loss_q_global, model.global_part.parameters(), create_graph=True)
                for g in range(len(task_grad_test_global)):
                    if args.re_w:
                        meta_grad_global[g] += task_grad_test_global[g].detach()*gender_w[i]
                    else:
                        meta_grad_global[g] += task_grad_test_global[g].detach()/float(args.tasks_per_metaupdate)
                if args.loss==1:
                    # loss_q_local = F.cross_entropy(logits_q, torch.squeeze(y_qry[i].long())-1) + adv*adv_loss
                    loss_q_local = adv_loss**args.adv_loss_power/args.adv_loss_power
                else:
                    # loss_q_local = F.mse_loss(logits_q, y_qry[i]) + adv*adv_loss
                    loss_q_local = adv_loss**args.adv_loss_power/args.adv_loss_power
                if args.con:
                    # now use coeff 1
                    loss_q_local += 1*F.mse_loss(output[2], logits_q)
                task_grad_test_local = torch.autograd.grad(loss_q_local, model.local_part.parameters(), create_graph=True)#, create_graph=True
                for g in range(len(task_grad_test_local)):
                    if args.re_w:
                        meta_grad_local[g] += task_grad_test_local[g].detach()*gender_w[i]
                    else:
                        meta_grad_local[g] += task_grad_test_local[g].detach()/float(args.tasks_per_metaupdate)
                if args.loss==1:
                    loss_after.append(F.cross_entropy(logits_q, torch.squeeze(y_qry[i].long())-1).item())
                else:
                    loss_after.append(F.mse_loss(logits_q, y_qry[i]).item())                
            # -------------- meta update --------------
            
            meta_optimiser.zero_grad()
            # set gradients of parameters manually
            for c, param in enumerate(model.local_part.parameters()):
                param.grad = meta_grad_local[c]
                param.grad.data.clamp_(-10, 10)
            for c, param in enumerate(model.global_part.parameters()):
                param.grad = meta_grad_global[c]
                param.grad.data.clamp_(-10, 10)
            if args.outer==1:
                for c, param in enumerate(model.global_fix_part.parameters()):
                    param.grad = meta_grad_global_fix[c]
                    param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            #scheduler.step()            
            x_spt, y_spt, x_qry, y_qry = [],[],[],[]
            
            loss_pre = np.array(loss_pre)
            loss_after = np.array(loss_after)
            logger.train_loss.append(np.mean(loss_pre))
            logger.valid_loss.append(np.mean(loss_after))
            logger.train_conf.append(1.96*np.std(loss_pre, ddof=0)/np.sqrt(len(loss_pre)))
            logger.valid_conf.append(1.96*np.std(loss_after, ddof=0)/np.sqrt(len(loss_after)))
            logger.test_loss.append(0)
            logger.test_conf.append(0)
    
            # print current results
            logger.print_info(epoch, iter_counter, start_time)
            print("adv loss pre:{}, adv_loss_after:{}".format(np.mean(adv_loss_pre), np.mean(adv_loss_after)))
            logging.info("adv loss pre:{}, adv_loss_after:{}".format(np.mean(adv_loss_pre), np.mean(adv_loss_after)))
            start_time = time.time()
            
            iter_counter += 1
        if epoch % (2) == 0:
            print('saving model at iter', epoch)
            logging.info('saving model at iter'+str(epoch))
            logger.valid_model.append(copy.deepcopy(model))
            utils.save_obj(logger, path)
            logging.info('Valid Eval')
            print('Valid Eval')
            evaluate_test_adv(args, model, dataloader_valid)
            logging.info('Test Eval')
            print('Test Eval')
            evaluate_test_adv(args, model, dataloader_test)

    return logger, model


def fairness_metrics(group_item_pred, group_item_rate):
    # value absolute underestimation overestimation fairness 
    male_pred, female_pred = group_item_pred[0], group_item_pred[1]
    male_rate, female_rate = group_item_rate[0], group_item_rate[1]
    shared_items = list(set(male_pred.keys())&set(female_pred.keys()))
    sign_value_fairness_list = []
    value_fairness_list = []
    absolute_fairness_list = []
    underestimation_fairness_list = []
    overestimation_fairness_list = []
    for item in shared_items:
        de_male = np.mean(np.array(male_pred[item]) - np.array(male_rate[item]))
        de_female = np.mean(np.array(female_pred[item]) - np.array(female_rate[item]))
        sign_de_male = np.mean(np.sign(np.array(male_pred[item]) - np.array(male_rate[item])))
        sign_de_female = np.mean(np.sign(np.array(female_pred[item]) - np.array(female_rate[item])))
        sign_value_fairness_item = abs(sign_de_male - sign_de_female)
        value_fairness_item = abs(de_male - de_female)
        absolute_fairness_item = abs(abs(de_male) - abs(de_female))
        underestimation_fairness_item = abs(max(0, -de_male)-max(0, -de_female))
        overestimation_fairness_item = abs(max(0, de_male)-max(0, de_female))
        sign_value_fairness_list.append(sign_value_fairness_item)
        value_fairness_list.append(value_fairness_item)
        absolute_fairness_list.append(absolute_fairness_item)
        underestimation_fairness_list.append(underestimation_fairness_item)
        overestimation_fairness_list.append(overestimation_fairness_item)
    sign_value_fairness = np.mean(sign_value_fairness_list)
    value_fairness = np.mean(value_fairness_list)
    absolute_fairness = np.mean(absolute_fairness_list)
    underestimation_fairness = np.mean(underestimation_fairness_list)
    overestimation_fairness = np.mean(overestimation_fairness_list)
    # non-parity fairness
    male_preds, female_preds = [], []
    male_values, female_values = [], []
    male_abs, female_abs = [], []
    for i, preds in male_pred.items():
        male_preds.extend(preds)
        male_values.extend(np.array(male_pred[i])-np.array(male_rate[i]))
        male_abs.extend(abs(np.array(male_pred[i])-np.array(male_rate[i])))
        
        # male_preds.append(np.mean(preds))
    for i, preds in female_pred.items():
        female_preds.extend(preds)
        female_values.extend(np.array(female_pred[i])-np.array(female_rate[i]))
        female_abs.extend(abs(np.array(female_pred[i])-np.array(female_rate[i])))
        # female_preds.append(np.mean(preds))
    non_parity_fairness = abs(np.mean(male_preds)-np.mean(female_preds))
    user_value_fairness = abs(np.mean(male_values)-np.mean(female_values))
    user_abs_fairness = abs(np.mean(male_abs)-np.mean(female_abs))
    print(sign_value_fairness)
    print(np.mean(male_preds), np.mean(female_preds))
    print(value_fairness, absolute_fairness, underestimation_fairness, overestimation_fairness, non_parity_fairness, user_value_fairness, user_abs_fairness)
    logging.info(str(value_fairness)+" "+str(absolute_fairness)+" "+str(underestimation_fairness)+" "+str(overestimation_fairness)+" "+str(non_parity_fairness)+" "+str(user_value_fairness)+" "+str(user_abs_fairness))
    male_preds, female_preds = [], []
    male_values, female_values = [], []
    male_abs, female_abs = [], []
    for item in shared_items:
        male_preds.extend(male_pred[item])
        male_values.extend(np.array(male_pred[item])-np.array(male_rate[item]))
        male_abs.extend(abs(np.array(male_pred[item])-np.array(male_rate[item])))
        female_preds.extend(female_pred[item])
        female_values.extend(np.array(female_pred[item])-np.array(female_rate[item]))
        female_abs.extend(abs(np.array(female_pred[item])-np.array(female_rate[item])))
    shared_non_parity_fairness = abs(np.mean(male_preds)-np.mean(female_preds))
    shared_user_value_fairness = abs(np.mean(male_values)-np.mean(female_values))
    shared_user_abs_fairness = abs(np.mean(male_abs)-np.mean(female_abs))
    print("shared user fair", shared_non_parity_fairness, shared_user_value_fairness, shared_user_abs_fairness)
    logging.info("shared fair "+str(shared_non_parity_fairness)+" "+str(shared_user_value_fairness)+" "+str(shared_user_abs_fairness))

def ndcg(y_pred, y, topk=3):
    ele_idx = heapq.nlargest(topk, zip(y_pred, itertools.count()))
    pred_index = np.array([idx for ele, idx in ele_idx], dtype=np.intc)
    rel_pred = []
    for i in pred_index:
        rel_pred.append(y[i])
    rel = heapq.nlargest(topk, y)
    idcg = np.cumsum((np.power(rel, 2)-1) / np.log2(np.arange(2, topk + 2)))
    dcg = np.cumsum((np.power(rel_pred, 2)-1) / np.log2(np.arange(2, topk + 2)))
    ndcg = dcg/idcg
    return ndcg[-1]


def evaluate_test(args, model, dataloader):
    model.eval()
    loss_all = []
    ndcg_all = []
    con_loss_all = []
    group_item_pred = {0:collections.defaultdict(list), 1:collections.defaultdict(list)}
    group_item_rate = {0:collections.defaultdict(list), 1:collections.defaultdict(list)}
    group_perf = {0:[],1:[]}
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda()
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        test_items = batch[5].numpy()[0]
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                output = model(x_spt[i], y_spt[i], [x_spt[i]], [y_spt[i]])
                logits = output[0]
                if args.loss==1:
                    loss = F.cross_entropy(logits, torch.squeeze(y_spt[i].long())-1)
                else:
                    loss = F.mse_loss(logits, y_spt[i])
                if args.con:
                    # now use coeff 1
                    loss += 1*F.mse_loss(output[1], logits)
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]  
                    fast_parameters.append(weight.fast) 
            output = model(x_qry[i], y_qry[i], [x_qry[i]], [y_qry[i]])
            logits = output[0]
            # con eval
            con_logits = output[1]
            if args.loss==1:
                logits = torch.unsqueeze(torch.argmax(logits,dim=1),1)+1
                con_logits = torch.unsqueeze(torch.argmax(con_logits,dim=1),1)+1
            con_loss_all.append(F.l1_loss(con_logits.float(), logits.float()).item())

            perf_mae = F.l1_loss(y_qry[i], logits.float()).item()
            loss_all.append(perf_mae)
            gender = int(x_qry[i][0, 21].detach().cpu().numpy())
            # N x 1
            true_rate = y_qry[i].detach().cpu().numpy()[:,0]
            pred_rate = logits.detach().cpu().numpy()[:,0]
            perf_ndcg = ndcg(pred_rate, true_rate, args.topk)
            ndcg_all.append(perf_ndcg)
            for i,item in enumerate(test_items):
                group_item_pred[gender][item].append(pred_rate[i])
                group_item_rate[gender][item].append(true_rate[i])
            group_perf[gender].append(perf_mae)
    # if args.fm:
    #     fairness_metrics(group_item_pred, group_item_rate)
    
    print('MAE: {}, NDCG@{}: {}'.format(np.mean(loss_all), args.topk, np.mean(ndcg_all)))
    print("user group fair:", abs(np.mean(group_perf[0])- np.mean(group_perf[1])))
    print("cf:", np.mean(con_loss_all))
    logging.info('MAE: {}, NDCG@{}: {}'.format(np.mean(loss_all), args.topk, np.mean(ndcg_all)))
    logging.info("user group fair:"+str(abs(np.mean(group_perf[0])- np.mean(group_perf[1]))))
    logging.info("cf:"+str(np.mean(con_loss_all)))
   
def evaluate_test_adv(args, model, dataloader):
    model.eval()
    loss_all, ndcg_all = [], []
    con_loss_all = []
    group_item_pred = {0:collections.defaultdict(list), 1:collections.defaultdict(list)}
    group_item_rate = {0:collections.defaultdict(list), 1:collections.defaultdict(list)}
    group_perf = {0:[],1:[]}
    if args.train_adv:
        adv_now = model.task_adv[0]
        print(adv_now)
        logging.info(adv_now)
    else:
        adv_now = args.adv
    # only for inner_fc 1
    for weight in model.global_part.parameters():
        weight.fast = None
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda()
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        test_items = batch[5].numpy()[0]
        for i in range(x_spt.shape[0]):
            gender = x_qry[i].detach().cpu().numpy()[0, 21]
            if args.re_w:
                if args.dual:
                    if gender==0:
                        adv = adv_now[0]
                    elif gender==1:
                        adv = adv_now[1]
                else:
                    adv = adv_now
            else:
                if args.dual:
                    if gender==0:
                        adv = adv_now[0]
                    elif gender==1:
                        adv = adv_now[1]
                else:
                    adv = adv_now
            # -------------- inner update --------------
            fast_parameters = model.local_part.parameters()
            for weight in model.local_part.parameters():
                weight.fast = None
            # local max update
            fast_parameters_max = model.local_max_part.parameters()
            for weight in model.local_max_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                output = model(x_spt[i], y_spt[i], [x_spt[i]], [y_spt[i]])
                logits, adv_loss = output[0], output[1]
                # local max update
                if args.loss==1:
                    if args.disable_inner_max:
                        loss_max = F.cross_entropy(logits, torch.squeeze(y_spt[i].long())-1)
                    else:
                        loss_max = F.cross_entropy(logits, torch.squeeze(y_spt[i].long())-1) - adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                    # loss_max = - adv*adv_loss
                else:
                    if args.disable_inner_max:
                        loss_max = F.mse_loss(logits, y_spt[i])
                        # loss_max = -adv_loss
                    else:
                        loss_max = F.mse_loss(logits, y_spt[i]) - adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                    # loss_max = - adv*adv_loss
                if args.con:
                    # now use coeff 1
                    loss_max += 1*F.mse_loss(output[2], logits)
                grad_max = torch.autograd.grad(loss_max, fast_parameters_max, create_graph=True)   
                fast_parameters_max = []
                for k, weight in enumerate(model.local_max_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad_max[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad_max[k]  
                    fast_parameters_max.append(weight.fast)
                if not args.disable_inner_adv:
                    if args.loss==1:
                        loss = adv_loss**args.adv_loss_power/args.adv_loss_power
                    else:
                        loss = adv_loss**args.adv_loss_power/args.adv_loss_power
                    if args.con:
                        # now use coeff 1
                        loss += 1*F.mse_loss(output[2], logits)
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)                    
                    fast_parameters = []
                    for k, weight in enumerate(model.local_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]  
                        fast_parameters.append(weight.fast)  
            output = model(x_qry[i], y_qry[i], [x_qry[i]], [y_qry[i]])
            logits, adv_loss = output[0], output[1]
            # con eval
            con_logits = output[2]
            if args.loss==1:
                logits = torch.unsqueeze(torch.argmax(logits,dim=1),1)+1
                con_logits = torch.unsqueeze(torch.argmax(con_logits,dim=1),1)+1
            con_loss_all.append(F.l1_loss(con_logits.float(), logits.float()).item())

            perf_mae = F.l1_loss(y_qry[i], logits.float()).item()
            loss_all.append(perf_mae)
            gender = int(x_qry[i][0, 21].detach().cpu().numpy())
            true_rate = y_qry[i].detach().cpu().numpy()[:,0]
            pred_rate = logits.detach().cpu().numpy()[:,0]
            perf_ndcg = ndcg(pred_rate, true_rate, args.topk)
            ndcg_all.append(perf_ndcg)
            for i,item in enumerate(test_items):
                group_item_pred[gender][item].append(pred_rate[i])
                group_item_rate[gender][item].append(true_rate[i])
            group_perf[gender].append(perf_mae)
    loss_all = np.array(loss_all)
    # if args.fm:
    #     fairness_metrics(group_item_pred, group_item_rate)
    print('MAE: {}, NDCG@{}: {}'.format(np.mean(loss_all), args.topk, np.mean(ndcg_all)))
    print("user group fair:", abs(np.mean(group_perf[0])- np.mean(group_perf[1])))
    print("cf:", np.mean(con_loss_all))
    logging.info('MAE: {}, NDCG@{}: {}'.format(np.mean(loss_all), args.topk, np.mean(ndcg_all)))
    logging.info("user group fair:"+str(abs(np.mean(group_perf[0])- np.mean(group_perf[1]))))
    logging.info("cf:"+str(np.mean(con_loss_all)))

def get_user_embedding_adv(args, model, dataloader, user_index):
    user_embedding = np.zeros((len(user_index), args.first_fc_hidden_dim//2))
    if args.train_adv:
        adv_now = model.task_adv[0]
        print(adv_now)
    else:
        adv_now = args.adv
    # only for inner_fc 1
    for weight in model.global_part.parameters():
            weight.fast = None
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda() # Batch_size(1) x 交互物品个数 x features
        y_spt = batch[1].cuda() 
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        user_id = batch[4][0]
        index = int(np.argwhere(user_index==user_id))
        for i in range(x_spt.shape[0]):
            gender = x_qry[i].detach().cpu().numpy()[0, 21]
            if args.re_w:
                if args.dual:
                    if gender==0:
                        adv = adv_now[0]
                    elif gender==1:
                        adv = adv_now[1]
                else:
                    adv = adv_now
            else:
                if args.dual:
                    if gender==0:
                        adv = adv_now[0]
                    elif gender==1:
                        adv = adv_now[1]
                else:
                    adv = adv_now
            # -------------- inner update --------------
            fast_parameters = model.local_part.parameters()
            for weight in model.local_part.parameters():
                weight.fast = None
            # local max update
            fast_parameters_max = model.local_max_part.parameters()
            for weight in model.local_max_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                output = model(x_spt[i], y_spt[i], [x_spt[i]], [y_spt[i]])
                logits, adv_loss = output[0], output[1]
                # local max update
                if args.loss==1:
                    if args.disable_inner_max:
                        loss_max = F.cross_entropy(logits, torch.unsqueeze(y_spt[i]).long()-1)
                    else:
                        loss_max = F.cross_entropy(logits, torch.unsqueeze(y_spt[i]).long()-1)- adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                else:
                    if args.disable_inner_max:
                        loss_max = F.mse_loss(logits, y_spt[i])
                    else:
                        loss_max = F.mse_loss(logits, y_spt[i])- adv*adv_loss**args.adv_loss_power/args.adv_loss_power
                grad_max = torch.autograd.grad(loss_max, fast_parameters_max, create_graph=True)   
                fast_parameters_max = []
                for k, weight in enumerate(model.local_max_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad_max[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad_max[k]  
                    fast_parameters_max.append(weight.fast)
            gender_idx = x_spt[0][:, 21]
            age_idx = x_spt[0][:, 20]
            occupation_idx = x_spt[0][:, 22]
            area_idx = x_spt[0][:, 23]
            # number of interactions x dim
            user_emb = F.relu(model.fc_user(model.user_emb(gender_idx, age_idx, occupation_idx, area_idx))).detach().cpu().numpy()
            user_embedding[index] = user_emb[0,:]
    return user_embedding

def get_user_embedding(args, model, dataloader, user_index):
    user_embedding = np.zeros((len(user_index), args.first_fc_hidden_dim//2))
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda() # Batch_size(1) x 交互物品个数 x features
        y_spt = batch[1].cuda() 
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        user_id = batch[4][0]
        index = int(np.argwhere(user_index==user_id))
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                # if args.adv:
                #     logits, adv_loss = model(x_spt[i])
                # else:
                output = model(x_spt[i], y_spt[i], [x_spt[i]], [y_spt[i]])
                logits = output[0]
                if args.loss==1:
                    loss = F.cross_entropy(logits, torch.unsqueeze(y_spt[i]).long()-1)
                else:
                    loss = F.mse_loss(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]  
                    fast_parameters.append(weight.fast)
            gender_idx = x_spt[0][:, 21]
            age_idx = x_spt[0][:, 20]
            occupation_idx = x_spt[0][:, 22]
            area_idx = x_spt[0][:, 23]
            # number of interactions x dim
            user_emb = F.relu(model.fc_user(model.user_emb(gender_idx, age_idx, occupation_idx, area_idx))).detach().cpu().numpy()
            user_embedding[index] = user_emb[0,:]
    return user_embedding



def fair_fine(args, model, dataloader_fair_train, dataloader_fair_test):
    # user embedding is not fixed after training
    model.eval()
    dataloader_train = DataLoader(Metamovie(args),
                                     batch_size=1,num_workers=args.num_workers)
    dataloader_valid = DataLoader(Metamovie(args, partition='test', test_way='new_user_valid'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
    dataloader_test = DataLoader(Metamovie(args, partition='test', test_way='new_user_test'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
    user_index_train = np.array(dataloader_train.dataset.final_index)
    user_index_test = np.array(dataloader_test.dataset.final_index)
    if args.adv:
        train_user_embedding = get_user_embedding_adv(args, model, dataloader_train, user_index_train)
        test_user_embedding = get_user_embedding_adv(args, model, dataloader_test, user_index_test)
    else:
        train_user_embedding = get_user_embedding(args, model, dataloader_train, user_index_train)
        test_user_embedding = get_user_embedding(args, model, dataloader_test, user_index_test)

    dataloader_fair_train.dataset.user_embedding = train_user_embedding
    dataloader_fair_test.dataset.user_embedding = test_user_embedding

    classifier = fair_classifier(args).cuda()
    # pre_model.eval()
    # pre_model.fc_user.weight.fast = None
    # pre_model.fc_user.bias.fast = None
    optimizer = torch.optim.Adam(classifier.parameters(), args.fair_lr)
    for epoch in range(100):
        pred_list, prob_list, label_list = [], [], []
        for c, batch in tqdm(enumerate(dataloader_fair_train)):
            classifier.train()
            x = batch[0].cuda()
            user_emb = batch[1].cuda()
            optimizer.zero_grad()
            prob, gender_idx, loss = classifier(x, user_emb)
            loss.backward()
            optimizer.step()

        classifier.eval()
        for c, batch in enumerate(dataloader_fair_test):
            x = batch[0].cuda()
            user_emb = batch[1].cuda()
            prob, gender_idx, loss = classifier(x, user_emb)
            prob = prob.detach().cpu().numpy()
            label = gender_idx.detach().cpu().numpy()
            label_list.extend(label)
            # acc
            pred = np.argmax(prob, axis=1)
            pred_list.extend(pred)
            # auc
            prob = prob[:,1]
            prob_list.extend(prob)
        acc = accuracy_score(label_list, pred_list)
        auc = roc_auc_score(label_list, prob_list)
        # print("Epoch {} AUC: {}".format(epoch, auc))
        print("Epoch {} Acc: {}, AUC: {}".format(epoch, acc, auc))
       

if __name__ == '__main__':
    args = parse_args()
    mode_path = utils.get_path_from_args(args)
    logging.basicConfig(filename="log/"+mode_path+".log")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    train_data = Metamovie(args, partition='train')
    global n_male_users, n_female_users
    n_male_users, n_female_users = 10, 10
    if not args.fair:
        if not args.test:
            if args.adv:
                run_adv(args, num_workers=1, log_interval=100, verbose=True, save_path=None)
            else:
                run(args, num_workers=1, log_interval=100, verbose=True, save_path=None)
        else:
            utils.set_seed(args.seed)
            code_root = os.path.dirname(os.path.realpath(__file__))
            # args.test = False
            mode_path = utils.get_path_from_args(args)
            # args.test = True
            path = '{}/{}_result_files/'.format(code_root, args.task) + mode_path
            logger = utils.load_obj(path)
            model = logger.valid_model[-1].cuda()
            dataloader_valid = DataLoader(Metamovie(args, partition='test', test_way='new_user_valid'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
            dataloader_test = DataLoader(Metamovie(args, partition='test', test_way='new_user_test'),#old, new_user, new_item, new_item_user
                                        batch_size=1, num_workers=args.num_workers)
            if args.adv:
                evaluate_test_adv(args, model, dataloader_test)
            else:
                evaluate_test(args, model, dataloader_test)
    else:
        utils.set_seed(args.seed)
        code_root = os.path.dirname(os.path.realpath(__file__))
        # args.fair = False
        mode_path = utils.get_path_from_args(args)
        # args.fair = True
        path = '{}/{}_result_files/'.format(code_root, args.task) + mode_path
        logger = utils.load_obj(path)
        pre_model = logger.valid_model[-1].cuda()
        dataloader_fair_train = DataLoader(Metamovie_fair(args, partition='train'),#old, new_user, new_item, new_item_user
                                        batch_size=128, num_workers=args.num_workers)
        dataloader_fair_valid = DataLoader(Metamovie_fair(args, partition='test', test_way='new_user_valid'),#old, new_user, new_item, new_item_user
                                        batch_size=128, num_workers=args.num_workers)
        dataloader_fair_test = DataLoader(Metamovie_fair(args, partition='test', test_way='new_user_test'),#old, new_user, new_item, new_item_user
                                        batch_size=128, num_workers=args.num_workers)
        fair_fine(args, pre_model, dataloader_fair_train, dataloader_fair_test)


