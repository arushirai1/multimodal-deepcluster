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

alphas=[]

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
    parser.add_argument('--cliplen', type=int, default=16,
                        help='clip len (default: 16)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="for contrastive learning")
    parser.add_argument('--contrastive', action='store_true', help='contrastive learning???')
    parser.add_argument('--gradual', action='store_true', help='gradual alpha rate')

    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--video_resume', default='', type=str, metavar='VPATH',
                        help='path to video checkpoint (default: None)')
    parser.add_argument('--text_resume', default='', type=str, metavar='TPATH',
                        help='path to text checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--ucf',type=int, default=1, help='using ucf101 dataset?')

    return parser.parse_args()

def _load_model(path, model, args, opt):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        args.start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict'].copy()
        to_delete = []
        # remove top_layer parameters from checkpoint
        for key in state_dict:
            if 'top_layer' in key:
                to_delete.append(key)
        for key in to_delete:
            del state_dict[key]
        model.load_state_dict(state_dict)
        opt.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
    return model, opt


def main(args):
    root, dictionary_pickle, metadata_path = build_paths()
    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)
    text_deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # video model
    video_model = InceptionI3d(400, in_channels=3)
    video_model.load_state_dict(torch.load('./pytorch_i3d/models/rgb_imagenet.pt'))
    video_model.replace_logits(args.nmb_cluster)
    video_model = torch.nn.DataParallel(video_model)

    # text model
    text_model = models.__dict__[args.arch](sobel=args.sobel, out=args.nmb_cluster)
    fd = int(text_model.top_layer.weight.size()[1])

    video_model.cuda()
    text_model.cuda()

    cudnn.benchmark = True

    # create optimizer
    video_optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, video_model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    text_optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, text_model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    video_criterion = nn.CrossEntropyLoss().cuda()
    text_criterion = nn.CrossEntropyLoss().cuda()
    contrastive_criterion = nn.TripletMarginLoss()

    # optionally resume from a checkpoint
    if args.video_resume:
        video_model, video_optimizer = _load_model(args.video_resume, video_model, args, video_optimizer)
    if args.text_resume:
        text_model.top_layer = None
        text_model, text_optimizer = _load_model(args.text_resume, text_model, args, text_optimizer)

    # creating checkpoint repo
    text_exp_check = os.path.join(args.exp, 'text-checkpoints')
    video_exp_check = os.path.join(args.exp, 'video-checkpoints')
    print("Exps: ", text_exp_check, video_exp_check)
    if not os.path.isdir(video_exp_check):
        os.makedirs(video_exp_check)
    if not os.path.isdir(text_exp_check):
        os.makedirs(text_exp_check)

    # creating cluster assignments log
    video_cluster_log = Logger(os.path.join(args.exp, 'video_clusters'))
    text_cluster_log = Logger(os.path.join(args.exp, 'text_clusters'))


    print('\n==> Preparing Data...\n')

    dataset = COIN(root, dictionary_pickle, metadata_path, train=True, method=args.modal, clip_len=args.cliplen, do_crop=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, pin_memory=False,  shuffle=False, num_workers=args.workers, timeout=500, drop_last=True)

    print('Number of Training Videos: %d' % len(dataset))

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        video_model.module.logits = None
        text_model.top_layer = None
        text_model.classifier = nn.Sequential(*list(text_model.classifier.children())[:-2])


        # get the features for the whole dataset
        features, text_features = compute_features(dataloader, video_model, text_model, len(dataloader)*args.batch)
        # cluster the features
        if args.verbose:
            print('Cluster the features')
        video_clustering_loss = deepcluster.cluster(features, verbose=args.verbose)
        #.set_trace()
        text_clustering_loss = text_deepcluster.cluster(text_features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        # will need edits for XDC, need to edit format of train_dataset
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  dataset,
                                                  text_deepcluster.images_lists, args.contrastive)
        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(text_model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        mlp.append(nn.Dropout(0.5).cuda())
        text_model.classifier = nn.Sequential(*mlp)
        text_model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        text_model.top_layer.weight.data.normal_(0, 0.01)
        text_model.top_layer.bias.data.zero_()
        text_model.top_layer.cuda()

        video_model.module.replace_logits(args.nmb_cluster)
        video_model.module.logits.conv3d.weight = nn.init.kaiming_normal_(video_model.module.logits.conv3d.weight, mode='fan_out')
        if video_model.module.logits.conv3d.bias is not None: video_model.module.logits.conv3d.bias.data.zero_()
        video_model.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        if args.contrastive:
            alphas=None
            if args.gradual:
                alphas=[0,0,0, 0.2, 0.2, 0.4, 0.4, 0.5]
            video_loss, text_loss = contrastive_train(train_dataloader, video_model, text_model, video_criterion, text_criterion, contrastive_criterion, video_optimizer, text_optimizer, epoch, alphas)
        else:
            video_loss, text_loss = train(train_dataloader, video_model, text_model, video_criterion, text_criterion, video_optimizer, text_optimizer, epoch)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Video Clustering loss: {2:.3f} \n'
                  'Text Clustering loss: {2:.3f} \n'
                  'Video i3d loss: {3:.3f}\n'
                  'Text model loss: {3:.3f}'
                  .format(epoch, time.time() - end, video_clustering_loss, text_clustering_loss, video_loss, text_loss))
            try:
                video_nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(video_cluster_log.data[-1])
                )

                text_nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(text_deepcluster.images_lists),
                    clustering.arrange_clustering(text_cluster_log.data[-1])
                )
                print('Video NMI against previous assignment: {0:.3f}'.format(video_nmi))
                print('Text NMI against previous assignment: {0:.3f}'.format(text_nmi))

            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': 'i3d',
                    'state_dict': video_model.state_dict(),
                    'optimizer' : video_optimizer.state_dict()},
                   os.path.join(video_exp_check, 'checkpoint.pth.tar'))

        torch.save({'epoch': epoch + 1,
                    'arch': 'roberta_model',
                    'state_dict': text_model.state_dict(),
                    'optimizer' : text_optimizer.state_dict()},
                   os.path.join(text_exp_check, 'checkpoint.pth.tar'))

        # save cluster assignments
        video_cluster_log.log(deepcluster.images_lists)
        text_cluster_log.log(text_deepcluster.images_lists)


def train(loader, video_model, text_model, video_crit, text_crit, video_opt, text_opt, epoch):
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
    video_losses = AverageMeter()
    text_losses = AverageMeter()

    data_time = AverageMeter()

    # switch to train mode
    video_model.train()
    text_model.train()

    end = time.time()

    for i, (input_tensor, video_target, text, text_target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save video_checkpoint
        n = len(loader) * epoch + i

        video_target = torch.autograd.Variable(video_target.cuda(non_blocking=True))
        text_target = torch.autograd.Variable(text_target.cuda(non_blocking=True))
        text_var = torch.autograd.Variable(text.cuda())
        video_var = torch.autograd.Variable(input_tensor.cuda())  # , volatile=True)

        video_scores = video_model(video_var)
        text_scores = text_model(text_var)

        # XDC aspect
        # utilize the text pseudolabels to calc the loss for video and vice versa
        video_loss = video_crit(video_scores, text_target)
        text_loss = text_crit(text_scores, video_target)

        # record loss
        video_losses.update(video_loss.item(), input_tensor.size(0))
        text_losses.update(text_loss.item(), text.size(0))

        # compute gradient and do SGD step
        video_opt.zero_grad()
        text_opt.zero_grad()

        video_loss.backward()
        text_loss.backward()

        video_opt.step()
        text_opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 20) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Video Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Text Loss: {text_loss.val:.4f} ({text_loss.avg:.4f})\t'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=video_losses, text_loss= text_losses))

    return video_losses.avg, text_losses.avg


def contrastive_train(loader, video_model, text_model, video_crit, text_crit, contrastive_crit, video_opt, text_opt, epoch, alphas=None):
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
    video_losses = AverageMeter()
    text_losses = AverageMeter()

    data_time = AverageMeter()

    # switch to train mode
    video_model.train()
    text_model.train()

    end = time.time()

    for i, (input_tensor, video_target, text, text_target, negative_text) in enumerate(loader):
        data_time.update(time.time() - end)

        # save video_checkpoint
        n = len(loader) * epoch + i

        video_target = torch.autograd.Variable(video_target.cuda(non_blocking=True))
        text_target = torch.autograd.Variable(text_target.cuda(non_blocking=True))
        text_var = torch.autograd.Variable(text.cuda())
        negative_text_var = torch.autograd.Variable(negative_text.cuda())
        video_var = torch.autograd.Variable(input_tensor.cuda())  # , volatile=True)

        video_scores = video_model(video_var)
        text_scores = text_model(text_var)
        negative_text_scores =  text_model(negative_text_var)

        # XDC aspect
        # utilize the text pseudolabels to calc the loss for video and vice versa
        constrastive_loss = contrastive_crit(video_scores, text_scores, negative_text_scores)
        alpha=args.alpha
        if args.gradual:
            if epoch < len(alphas):
                alpha=alphas[epoch]
            else:
                alpha=alphas[-1]
        video_loss = (1-alpha)*constrastive_loss + alpha*video_crit(video_scores, video_target)
        text_loss = (1-alpha)*constrastive_loss + alpha*text_crit(text_scores, video_target)

        # record loss
        video_losses.update(video_loss.item(), input_tensor.size(0))
        text_losses.update(text_loss.item(), text.size(0))

        # compute gradient and do SGD step
        video_opt.zero_grad()
        text_opt.zero_grad()

        video_loss.backward(retain_graph=True)
        text_loss.backward()

        video_opt.step()
        text_opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 20) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Video Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Text Loss: {text_loss.val:.4f} ({text_loss.avg:.4f})\t'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=video_losses, text_loss= text_losses))

    return video_losses.avg, text_losses.avg

def compute_features(dataloader, video_model, text_model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    video_model.eval()
    text_model.eval()

    # discard the label information in the dataloader
    try:
        for i, (input_tensor, _, text) in enumerate(dataloader):
            torch.no_grad()

            try:
                text_aux = text_model.extract_features(text.cuda()).data.cpu().numpy()
                text_aux = text_aux.astype('float32')
                input_var = torch.autograd.Variable(input_tensor.cuda())
                aux = video_model.module.extract_features(input_var).data.cpu().numpy()
                aux = aux.astype('float32')

                if i == 0:
                    features = np.zeros((N, aux.shape[1]), dtype='float32')
                    text_features = np.zeros((N, text_aux.shape[1]), dtype='float32')
                if i < len(dataloader):
                    features[i * args.batch: (i + 1) * args.batch] = aux
                    text_features[i * args.batch: (i + 1) * args.batch] = text_aux
                else:
                    # special treatment for final batch
                    features[i * args.batch:] = aux
                    text_features[i * args.batch:] = text_aux

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.verbose and (i % 50) == 0:
                    print('{0} / {1}\t' 
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                          .format(i, len(dataloader), batch_time=batch_time))
            except Exception as e:
                print("RAM Usage: ", str(psutil.virtual_memory().percent))
                print(GPUInfo.gpu_usage())

                print("failed: ", e)
                return
    except RuntimeError:
        print("RAM Usage: ", str(psutil.virtual_memory().percent))
        print(GPUInfo.gpu_usage())

        return features, text_features
    except Exception as e:
        print("Error {}".format(e))
    finally:
        return features, text_features


if __name__ == '__main__':
    args = parse_args()
    print("Clip len: ", args.cliplen)
    print("Args: ", args)
    main(args)
