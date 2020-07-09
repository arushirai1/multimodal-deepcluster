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

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--arch', '-a', type=str,
                        choices=['alexnet', 'vgg16', 'c3d', 'roberta_model'], default='roberta_model',
                        help='Text architecture (default: roberta_model)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--modal', type=str, choices=['text_only', 'joint', 'video_only'],
                        default='text_only', help='modality to train on (default: text_only)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
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
    parser.add_argument('--ucf',type=int, default=1, help='using ucf101 dataset?')

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
    #clip_len = 16

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    if text_only:
        model = models.__dict__[args.arch](sobel=args.sobel, out=args.nmb_cluster)
        fd = int(model.top_layer.weight.size()[1])
    else:
        model = InceptionI3d(400, in_channels=3)
        model.load_state_dict(torch.load('./pytorch_i3d/models/rgb_imagenet.pt'))
        model.replace_logits(args.nmb_cluster)
    #model.top_layer = None
    model = torch.nn.DataParallel(model) #WARNING to test
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            state_dict=checkpoint['state_dict'].copy()
            to_delete=[]
            # remove top_layer parameters from checkpoint
            for key in state_dict:
                if 'top_layer' in key:
                    to_delete.append(key)
            for key in to_delete:
                del state_dict[key]
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

   # load the data
    end = time.time()
    '''
    
    if args.cifar:
        dataset = datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transforms.Compose(tra))
        part_tr = torch.utils.data.random_split(dataset, [1000, len(dataset) - 1000])[0]

    else:
        dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(part_tr,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)
    '''

    print('\n==> Preparing Data...\n')

    dataset = COIN(root, dictionary_pickle, metadata_path, train=True, method=args.modal, do_crop=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    '''
    trainset = UCF10(class_idxs=class_idxs, split=train_split, frames_root=frames_root,
                     clip_len=clip_len, train=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = UCF10(class_idxs=class_idxs, split=test_split, frames_root=frames_root,
                    clip_len=clip_len, train=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    '''

    print('Number of Training Videos: %d' % len(dataset))

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        if text_only:
            model.module.top_layer = None
            model.module.classifier = nn.Sequential(*list(model.module.classifier.children())[:-1])
        else:
            model.module.logits = None

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  dataset)
        print(train_dataset)
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
        if text_only:
            mlp = list(model.module.classifier.children())
            mlp.append(nn.ReLU(inplace=True).cuda())
            model.module.classifier = nn.Sequential(*mlp)
            model.module.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
            model.module.top_layer.weight.data.normal_(0, 0.01)
            model.module.top_layer.bias.data.zero_()
            model.module.top_layer.cuda()
        else:
            model.module.replace_logits(args.nmb_cluster)
            model.module.logits.conv3d.weight = nn.init.kaiming_normal_(model.module.logits.conv3d.weight, mode='fan_out')
            if model.module.logits.conv3d.bias is not None: model.module.logits.conv3d.bias.data.zero_()
            model.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)


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
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    '''
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )  
    '''

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
            print("Text", text[0][:10])
            input_var = ' '.join(text[0].split()[:512])#torch.autograd.Variable(text)#.cuda())  # , volatile=True)
        else:
            input_var = torch.autograd.Variable(input_tensor.cuda())  # , volatile=True)

        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        print(output.shape)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        #optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        #optimizer_tl.step()

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

    return losses.avg

def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _, text) in enumerate(dataloader):
        torch.no_grad()
        #print(i, text)

        if text_only:
            #text = torch.autograd.Variable(text)#.cuda())  # , volatile=True)
            aux = model.module.extract_features(text[0].split()[:512]).data.cpu().numpy()
        else:
            input_var = torch.autograd.Variable(input_tensor.cuda())  # , volatile=True)
            aux = model.module.extract_features(input_var).data.cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t' 
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    args = parse_args()
    global text_only
    text_only = args.modal in 'text_only'
    print(text_only)

    main(args)
