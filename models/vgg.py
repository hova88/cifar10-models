'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torchinfo import summary

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_head(512,10) #nn.Linear(512, 10)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_head(self, num_features, num_classes, pool_type='avg', use_conv=False):
        if pool_type == 'avg':
            pool = nn.AvgPool2d(kernel_size=1, stride=1)
        elif pool_type == 'max':
            pool = nn.MaxPool2d(kernel_size=1, stride=1)
        if num_classes <= 0:
            fc = nn.Identity()  # pass-through (no classifier)
        elif use_conv:
            fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
        else:
            fc = nn.Linear(num_features, num_classes, bias=True)
        flatten = nn.Flatten(1) #if use_conv and pool_type else nn.Identity()
        layers = [pool , flatten , fc]
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = VGG('VGG13')
    x = torch.randn(2,3,32,32)
    y = model(x)
    summary(model , (2,3,32,32),
        verbose=1,
        col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"])
