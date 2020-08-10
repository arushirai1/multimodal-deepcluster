#inception network with 512 linear layer and than concat this with joint model
import torch
import torch.nn as nn
from .roBERTa import *
from .pytorch_i3d import *
import torch.nn.functional as F

__all__ = ['JointModel', 'joint']


class JointModel(nn.Module):
    def __init__(self, num_classes, path_to_i3d_weights):
        super(JointModel, self).__init__()
        self.i3d=InceptionI3d(400, in_channels=3)
        self.i3d.load_state_dict(torch.load(path_to_i3d_weights))
        self.i3d.replace_logits(512, F.relu)

        self.roberta=roberta_model(out=num_classes)
        self.roberta.top_layer=None
        self.top_layer=nn.Linear(1024, out_features=num_classes)

    def forward(self, video, text):
        x_vid = self.i3d(video)
        x_text = self.roberta(text)
        x=torch.cat([x_vid, x_text], dim=1)

        x = self.top_layer(x)
        return x
    def initialize_weights(self):
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()

    def extract_features(self, video, text):
        x_vid = self.i3d(video)
        x_text = self.roberta(text)
        x=torch.cat([x_vid, x_text], dim=1)
        return x


def joint(sobel=False, bn=True, out=10, pathto=''):
    return JointModel(out, pathto)