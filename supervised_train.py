# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time
import pdb

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import deepcluster.clustering as clustering
import deepcluster.models as models
from deepcluster.util import AverageMeter, Logger, UnifLabelSampler
from COIN_Dataset import COIN
from Utils import build_paths
from deepcluster.models.pytorch_i3d import InceptionI3d
import psutil
from gpuinfo import GPUInfo

from multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--arch', '-a', type=str,
                        choices=['alexnet', 'vgg16', 'c3d', 'joint', 'roberta_model'], default='roberta_model',
                        help='Text architecture (default: roberta_model)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--modal', type=str, choices=['text_only', 'joint', 'video_only'],
                        default='text_only', help='modality to train on (default: text_only)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--cliplen', type=int, default=64,
                        help='clip len (default: 16)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--ucf', type=int, default=1, help='using ucf101 dataset?')

    return parser.parse_args()


def main(args):
    '''
    best_acc = 0                                                           # best test accuracy
    start_epoch = 0                                                        # start from epoch 0 or last checkpoint epoch
    initial_lr = .00001
    batch_size = 12
    num_workers = 2
    :param args:
    :return:
    '''
    root, dictionary_pickle, metadata_path = build_paths()
    # clip_len = 16

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    dataset = COIN(root, dictionary_pickle, metadata_path, train=True, method=args.modal, clip_len=args.cliplen,
                   do_crop=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, pin_memory=False, shuffle=False,
                                             num_workers=args.workers, timeout=500, drop_last=True)
    val_dataset = COIN(root, dictionary_pickle, metadata_path, method=args.modal, clip_len=args.cliplen, train=False, do_crop=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(args.batch/2),
                                             shuffle=False,
                                             num_workers=args.workers)

    num_classes = len(dataset.class_dict)

    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load('./pytorch_i3d/models/rgb_imagenet.pt'))
    model.replace_logits(num_classes)
    model = torch.nn.DataParallel(model)

    # model.top_layer = None
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)


    print('\n==> Preparing Data...\n')


    print('Number of Training Videos: %d' % len(dataset))


    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(dataloader, model, criterion, optimizer, epoch)
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))
        prec1, prec5, loss = validate(val_loader, model, criterion)
        print("Validation [%d]: " %epoch, prec1, prec5, loss)

def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_tensor, target, text) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict()
            }, path)

        target = target.cuda(non_blocking=True)
        if text_only:
            input_var = text.cuda()  # ' '.join(text[0].split()[:512])#torch.autograd.Variable(text)#.cuda())  # , volatile=True)
        elif is_joint:
            text = text.cuda()
            input_var = torch.autograd.Variable(input_tensor.cuda())  # , volatile=True)
        else:
            input_var = torch.autograd.Variable(input_tensor.cuda())  # , volatile=True)

        target_var = torch.autograd.Variable(target)

        if is_joint:
            output = model(input_var, text)
        else:
            output = model(input_var)
        loss = crit(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input_tensor.size(0))
        top5.update(prec5[0], input_tensor.size(0))
        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        # optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        # optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))
    print("Train [%d]: " % epoch, top1.avg, top5.avg, losses.avg)
    return losses.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    for i, (input_tensor, target, text) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            if args.modal == 'text_only':
                text_var = torch.autograd.Variable(text.cuda())
            elif args.modal =='video_only':
                input_var = torch.autograd.Variable(input_tensor.cuda())
            else:
                text_var = torch.autograd.Variable(text.cuda())
                input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)

        if args.modal == 'text_only':
            output = model(text_var)  # reglog(output)
        elif args.modal =='video_only':
            output = model(input_var)
        else:
            output = model(input_var, text_var)

        output_central = output

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input_tensor.size(0))
        top5.update(prec5[0], input_tensor.size(0))
        loss = criterion(output_central, target_var)
        losses.update(loss.item(), input_tensor.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 10 == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    args = parse_args()
    global text_only
    text_only = args.modal in 'text_only'
    global is_joint
    is_joint = args.modal in 'joint'
    print("Is text only? ", text_only)
    print("Clip len: ", args.cliplen)

    main(args)
