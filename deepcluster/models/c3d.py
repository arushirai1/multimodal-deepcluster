import torch
import torch.nn as nn
from torchsummary import summary

__all__ = [ 'C3D', 'c3d']

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained_path='', pretrained=False):

        self.pretrained_path = pretrained_path

        super(C3D, self).__init__()

        self.features = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                                      nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                                      nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                                      nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                                      nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        self.classifier = nn.Sequential(*[nn.Linear(8192, 4096),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(0.5),
                                          nn.Linear(4096, 4096),
                                          nn.ReLU(inplace=True)])
        self.top_layer = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()



    def forward(self, x):

        x = self.features(x)

        x = x.view(-1, 8192)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name1 = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        corresp_name = {
                        # Conv1
                        "conv1.weight": "conv1.weight",
                        "conv1.bias": "conv1.bias",
                        # Conv2
                        "conv2.weight": "conv2.weight",
                        "conv2.bias": "conv2.bias",
                        # Conv3a
                        "conv3a.weight": "conv3a.weight",
                        "conv3a.bias": "conv3a.bias",
                        # Conv3b
                        "conv3b.weight": "conv3b.weight",
                        "conv3b.bias": "conv3b.bias",
                        # Conv4a
                        "conv4a.weight": "conv4a.weight",
                        "conv4a.bias": "conv4a.bias",
                        # Conv4b
                        "conv4b.weight": "conv4b.weight",
                        "conv4b.bias": "conv4b.bias",
                        # Conv5a
                        "conv5a.weight": "conv5a.weight",
                        "conv5a.bias": "conv5a.bias",
                        # Conv5b
                        "conv5b.weight": "conv5b.weight",
                        "conv5b.bias": "conv5b.bias",
                        # fc6
                        "fc6.weight": "fc6.weight",
                        "fc6.bias": "fc6.bias",
                        # fc7
                        "fc7.weight": "fc7.weight",
                        "fc7.bias": "fc7.bias",
                        }

        p_dict = torch.load(self.pretrained_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

def c3d(sobel=False, bn=True, out=400):
    model = C3D(out)
    return model
