# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np
import torch
import torch.nn as nn

__all__ = ['RobertaCaptions', 'roberta_model']


class RobertaCaptions(nn.Module):
    def __init__(self, out=128):
        super(RobertaCaptions, self).__init__()
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').cuda()
        self.classifier = nn.Sequential(nn.Linear(1024, 512, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)).cuda()
        self.top_layer=nn.Linear(512, out).cuda()

        self._initialize_weights()

    def forward(self, x):
        tokens = self.roberta.encode(x)
        x = self.roberta.extract_features(tokens)
        pool= nn.AdaptiveAvgPool2d((1,1024)) #nn.MaxPool2d((x.shape[1], 1))
        x=pool(x).squeeze(1)
        x=self.classifier(x)
        return self.top_layer(x) #self.pool(x.permute(0, 2, 1))  # .cuda()).detach().cpu()

    def extract_features(self, x):
        tokens = self.roberta.encode(x)
        x = self.roberta.extract_features(tokens)
        pool= nn.AdaptiveAvgPool2d((1,1024)) #nn.MaxPool2d((x.shape[1], 1))
        x=pool(x).squeeze(1)
        return x

    def _initialize_weights(self):
        self.classifier[0].weight.data.normal_(0, 0.01)
        self.classifier[0].bias.data.zero_()
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()

def roberta_model(sobel=False, bn=True, out=1000):
    model = RobertaCaptions(out)
    return model
