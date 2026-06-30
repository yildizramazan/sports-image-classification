# Custom CNN model architecture definition
import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=7, num_blocks=4, base_filters=32, dropout_rate=0.3):
        
        super(CustomCNN, self).__init__()
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # feature extraction layers
        layers = []
        in_channels = 3
        
        for i in range(num_blocks):
            out_channels = base_filters * (2**i) # 32, 64, 128...

            layers.extend([

                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),

                nn.BatchNorm2d(out_channels),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2),

            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        
        # classifiers
        final_features_channels = base_filters * (2 ** (num_blocks - 1))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return total, trainable, frozen
        

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


        
