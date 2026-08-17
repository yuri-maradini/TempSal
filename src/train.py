import argparse
import os
import torch
import sys
import time
import wandb
import torch.nn as nn
from tqdm import tqdm
from dataloader import SaliconDataset
from loss import *
from utils import AverageMeter
from utils import img_save
from torchvision import utils
import torch.nn.functional as nnf
from os.path import join
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=30, type=int)
parser.add_argument('--lr',default=1e-5, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--lr_sched',default=False, type=bool)
parser.add_argument('--dilation',default=False, type=bool)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--optim',default="Adam", type=str)

parser.add_argument('--load_weight',default=1, type=int)
parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=-1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--vol_loss_coeff',default=1.0, type=float)
parser.add_argument('--train_enc',default=1, type=int)

parser.add_argument('--dataset_dir',default="../data/", type=str)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--log_interval',default=60, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--train_model',default=False, type=bool)
parser.add_argument('--time_slices',default=5, type=int)
parser.add_argument('--selected_slices',default="", type=str)
parser.add_argument('--results_dir',default="", type=str )

# Path to save the model weights
parser.add_argument('--model_val_path',default="model.pt", type=str)
# If the model type is pnas_boosted, specify the path of the pre-trained pnas model here
parser.add_argument('--model_path',default="", type=str)
# If the model type is pnas_boosted, specify the path of the pre-trained pnasvol model here
parser.add_argument('--model_vol_path',default="", type=str)


args = parser.parse_args()

# No wandb.init() call existed before, so every wandb.log() below raised
# immediately ("You must call wandb.init() before wandb.log()") -- true for
# any run, not specific to the UEyes/temporal-volume changes. Set
# WANDB_MODE=disabled (or =offline) in the environment to run without a
# wandb account/API key, e.g. for local smoke tests.
wandb.init(project="tempsal-ueyes", config=vars(args))

train_img_dir = args.dataset_dir + "images/train/"
train_gt_dir = args.dataset_dir + "maps/train/"
train_fix_dir = args.dataset_dir + "fixation_maps/train/"
train_vol_dir = args.dataset_dir + "saliency_volumes_5/train/"

val_img_dir = args.dataset_dir + "images/val/"
val_gt_dir = args.dataset_dir + "maps/val/"
val_fix_dir = args.dataset_dir + "fixation_maps/val/"
val_vol_dir = args.dataset_dir + "saliency_volumes_5/val/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_vol = args.enc_model == "pnas_boosted_multi"

if args.enc_model == "pnas":
    print("PNAS Model")
    from model import PNASModel
    model = PNASModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

elif args.enc_model == "pnas_boosted_multi":
    print("PNAS Boosted Model PNASBoostedModelMultiLevel")
    from model import PNASBoostedModelMultiLevel
    model = PNASBoostedModelMultiLevel(device, args.model_path, args.model_vol_path, args.time_slices, train_model=args.train_model,selected_slices = args.selected_slices )


if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
model.to(device)


train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids,
                                vol_dir=(train_vol_dir if use_vol else None), time_slices=args.time_slices)
val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids,
                              vol_dir=(val_vol_dir if use_vol else None), time_slices=args.time_slices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.no_workers)



def loss_func(pred_map, gt, fixations, args):
    loss = torch.zeros(1, device=pred_map.device)
    criterion = nn.L1Loss()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.nss:
        loss += args.nss_coeff * nss(pred_map, fixations)
    if args.l1:
        loss += args.l1_coeff * criterion(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)
    #print("Loss: ", loss)
    return loss

def vol_loss_func(vol_pred, vol_gt, args):
    # Per-slice version of loss_func's kldiv/cc terms, applied independently
    # to each of the time_slices temporal maps. NSS/SIM/L1 are left out here:
    # NSS would need the per-slice binary fixation_volumes_5 (generated in
    # Step 2 but not currently wired into the loader), and SIM/L1 are off by
    # default for the aggregate loss too, so this stays consistent with the
    # default loss configuration rather than an oversight.
    loss = torch.zeros(1, device=vol_pred.device)
    for t in range(vol_pred.size(1)):
        if args.kldiv:
            loss += args.kldiv_coeff * kldiv(vol_pred[:, t], vol_gt[:, t])
        if args.cc:
            loss += args.cc_coeff * cc(vol_pred[:, t], vol_gt[:, t])
    return loss / vol_pred.size(1)

def train(model, optimizer, loader, epoch, device, args, use_vol):
    model.train()
    if use_vol:
        # requires_grad=False on pnas_sal only stops its weights/biases from
        # being updated by the optimizer -- it does NOT stop BatchNorm's
        # running_mean/running_var buffers from drifting on every forward
        # pass while the module is in train() mode, which model.train()
        # above just put it in. Force it back to eval() so the "frozen"
        # branch is actually frozen, statistics included, not just its
        # learnable weights.
        base_model = model.module if hasattr(model, 'module') else model
        base_model.pnas_sal.eval()

    tic = time.time()

    total_loss = 0.0
    cur_loss = 0.0

    for idx, batch in enumerate(loader):
        if use_vol:
            img, gt, fixations, vol = batch
            vol = vol.to(device)
        else:
            img, gt, fixations = batch
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()
        pred_map, vol_pred = model(img)

        assert pred_map.size() == gt.size()

        loss = loss_func(pred_map, gt, fixations, args)
        if use_vol:
            loss = loss + args.vol_loss_coeff * vol_loss_func(vol_pred, vol, args)
        loss.backward()

        total_loss += loss.item()
        cur_loss += loss.item()

        optimizer.step()
        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
            wandb.log({"loss": cur_loss/args.log_interval})
            cur_loss = 0.0
            sys.stdout.flush()

    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)

def validate(model, loader, epoch, device, args, use_vol):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    vol_cc_loss = AverageMeter()
    vol_kldiv_loss = AverageMeter()

    for batch in tqdm(loader):
        if use_vol:
            img, gt, fixations, vol = batch
            vol = vol.to(device)
        else:
            img, gt, fixations = batch
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map   , vol_pred = model(img)

        cc_loss.update(cc(pred_map, gt))
        kldiv_loss.update(kldiv(pred_map, gt))
        nss_loss.update(nss(pred_map, fixations))
        sim_loss.update(similarity(pred_map, gt))

        if use_vol:
            for t in range(vol_pred.size(1)):
                vol_cc_loss.update(cc(vol_pred[:, t], vol[:, t]))
                vol_kldiv_loss.update(kldiv(vol_pred[:, t], vol[:, t]))

    print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
    log_dict = {"CC": cc_loss.avg, 'KLDIV': kldiv_loss.avg, 'NSS': nss_loss.avg, 'SIM': sim_loss.avg}
    if use_vol:
        print('[{:2d},   val] Vol CC : {:.5f}, Vol KLDIV : {:.5f}'.format(epoch, vol_cc_loss.avg, vol_kldiv_loss.avg))
        log_dict.update({'Vol/CC': vol_cc_loss.avg, 'Vol/KLDIV': vol_kldiv_loss.avg})
    wandb.log(log_dict)
    sys.stdout.flush()

    return cc_loss.avg,cc_loss,kldiv_loss,nss_loss,sim_loss

params = list(filter(lambda p: p.requires_grad, model.parameters()))

if args.optim=="Adam":
    optimizer = torch.optim.Adam(params, lr=args.lr)
if args.optim=="Adagrad":
    optimizer = torch.optim.Adagrad(params, lr=args.lr)
if args.optim=="SGD":
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
if args.lr_sched:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

print(device)
best_loss = 0
for epoch in range(0, args.no_epochs):
    loss = train(model, optimizer, train_loader, epoch, device, args, use_vol)

    with torch.no_grad():
                cc_loss,cc_loss_obj,kldiv_loss,nss_loss,sim_loss = validate(model, val_loader, epoch, device, args, use_vol)
                cc_loss -=kldiv_loss.avg
                if epoch == 0 :
                    best_loss = cc_loss
                if best_loss <= cc_loss:
                    best_loss = cc_loss
                    print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
                    wandb.log({"Best/CC mean": cc_loss,"Best/CC median": cc_loss_obj.get_median(), "Best/CC std": cc_loss_obj.get_std(),
        "Best/KLD mean": kldiv_loss.avg,"Best/KLD median": kldiv_loss.get_median(), "Best/KLD std": kldiv_loss.get_std(),
        "Best/NSS mean": nss_loss.avg,"Best/NSS median": nss_loss.get_median(), "Best/NSS std": nss_loss.get_std(),
        "Best/SIM mean": sim_loss.avg,"Best/SIM median": sim_loss.get_median(), "Best/SIM std": sim_loss.get_std()})
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.model_val_path)
                    else:
                        torch.save(model.state_dict(), args.model_val_path)
                print()

    if args.lr_sched:
        scheduler.step()
