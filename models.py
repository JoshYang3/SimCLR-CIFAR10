import torch.nn as nn


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(weights=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.classifier[-1].in_features # same as self.enc.classifier[-1], and add in_features
        #self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        
        ### VGG11 ### learning rate = 0.03
        self.enc.features[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.features[2] = nn.Identity()
        self.enc.classifier[-1] = nn.Identity() 
        
        '''
        ### VGG19
        self.enc.features[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.features[4] = nn.Identity()
        self.enc.classifier[-1] = nn.Identity() 
        
        
        ### Alexnet ### learning rate = 0.15
        self.enc.features[0] = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.features[2] = nn.Identity()
        self.enc.classifier[-1] = nn.Identity()  # remove final fully connected layer.
        

        ### Resnet ### learning rate = 0.6
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.
        '''
        
        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection



