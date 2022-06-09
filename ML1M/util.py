import hashlib
import os
import pickle
import random

import numpy as np
import torch


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    # args_str = str(args)
    # path = hashlib.md5(args_str.encode()).hexdigest()
    # path = "meta_adv_{}_xgender_{}_evalstep_{}".format(args.adv, args.remove_gender, args.num_grad_steps_eval)
    path = "both_adv_{}_xgender_{}_evalstep_{}".format(args.adv, args.remove_gender, args.num_grad_steps_eval)
    path = path+"_outer_{}".format(args.outer)
    if args.normalize:
        path = path+"_normalize_{}".format(args.normalize)
    if args.all_adv:
        path = path+"_alladv_{}".format(args.all_adv)
    if args.schedule:
        path = path+"_schedule_{}".format(args.schedule)
    if args.disable_inner_adv:
        path = path+"_x_inner_adv_{}".format(args.disable_inner_adv)
    if args.disable_inner_max:
        path = path+"_x_inner_max_{}".format(args.disable_inner_max)
    if args.adv_loss_power!=1:
        path = path+"_adv_loss_power_{}".format(args.adv_loss_power)
    if args.more!=1:
        path = path+"_more_{}".format(args.more)
    if args.loss:
        path = path+"_loss_{}".format(args.loss)
    if args.inner_fc:
        path = path+"_inner_fc_{}".format(args.inner_fc)
    if args.out:
        path = path+"_out_{}".format(args.out)
    if args.out2: # only out2=0
        path = path+"_out_{}".format(args.out2)
    if args.sim:
        path = path+"_sim_{}".format(args.sim)
    if args.item_adv:
        path = path+"_advitem_{}".format(args.item_adv)
    if args.adv2:
        path = path+"_adv2_{}".format(args.adv2)
    if args.train_adv:
        path = path+"_advtrain_{}".format(args.train_adv)
    if args.re_w:
        path = path+"_reweight_{}".format(args.re_w)
    if args.dual:
        path = path+"_dual_{}".format(args.dual)
    if args.lam:
        if args.group==0:
            raise ValueError("指定group loss")
        path = path+"_group_{}_{}".format(args.group, args.lam)
    if args.seed!=53:
        path = path+"_seed_{}".format(args.seed)
    if args.con:
        path = path+"_con_{}".format(args.con)
    print(path)
    return path


def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError('I dont know where I am; please specify a path for saving results.')
