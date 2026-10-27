import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F
from bimamba import Mamba, MambaConfig 
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from mtdc import MTDC
from lpg import LPG
from c2r import C2R

icc=64+2
ic=72
oc=128
ws=13
fs=(ws+1)//2

num_class=11

d_model=ws*ws

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convs=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_convs(x)

class PositionalEncoding(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.register_buffer('position_ids', torch.arange(height * width))
        self.embedding = nn.Embedding(height * width, channels)
        self.height = height
        self.width = width

    def forward(self, x):
        B, C, H, W = x.shape
        #assert H == self.height and W == self.width

        # [H*W, C]
        pos_embed = self.embedding(self.position_ids)
        # [1, C, H, W]
        pos_embed = pos_embed.view(1, C, H, W)
        # expand to batch
        pos_embed = pos_embed.expand(B, -1, -1, -1)
        return x + pos_embed    
    
class Pooling(nn.Module):
    """
    @article{ref-vit,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)

class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

def BasicConv(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),)

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=3,dim_feedforward=384,nhead=8,reverse=False,dropout=0.1):
        super(EncoderLayers, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_norm =LayerNorm(encoder_in)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.reverse=reverse

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        if self.reverse:
            x=torch.flip(x, dims=[1])

        x = self.encoder(x)

        if self.reverse:
            x=torch.flip(x, dims=[1])

        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x   

class MambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers,use_cuda=True))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class TransBlock(nn.Module):
    def __init__(self, ic: int, ws: int,num_encoder_layers=1):
        """
        Args:
            ic (int): input channels
            ws (int): width = height
        """
        super(TransBlock, self).__init__()
        self.ic = ic
        self.ws = ws  # window size or feature map size

        self.BN1 = nn.BatchNorm2d(ic)
        self.pos1 = PositionalEncoding(ic, ws, ws)
        self.attn1 = EncoderLayers(
            encoder_in=ic,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=ic*2,
            nhead=8
        )
    def forward(self, x):
        x = self.BN1(x)        
        x = self.pos1(x)        
        x = self.attn1(x)      
        return x

class PNC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PNC, self).__init__()
        self.pnc = nn.Sequential(
            BasicConv(in_channels,out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.pnc(x)

class StandardCA(nn.Module):
    def __init__(self, channels, reduction=8): 
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UpWOconv(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpWOconv,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.oneconv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        
        return self.oneconv(x)  

def MTDCConv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True):
    """
    Convolution wrapper used by the MTDC module.

    The ``bias`` argument is accepted for interface compatibility and ignored
    to match the behavior of ``BasicConv``.
    """
    return BasicConv(in_channels, out_channels, kernel_size, stride, padding)   
    
class LECMamba(torch.nn.Module):
    def __init__(self):
        super(LECMamba, self).__init__()
        
        self.conv0 = BasicConv(in_channels=icc, out_channels=ic, kernel_size=1, stride=1, padding=0) 
        #Block

        self.inc= DoubleConv(ic,oc)
        self.down1=PNC(oc,oc*2) 
        self.down2=PNC(oc*2,oc*4) 
        
        self.tb1=TransBlock(oc*2,fs-1,num_encoder_layers=3) 
        
        self.up2= UpWOconv(oc*2,oc) #UpWOconv(oc*2,oc)
        self.up1= UpWOconv(oc*4,oc*2)
        self.tbU2=TransBlock(oc,ws,num_encoder_layers=3) 

        self.dropout=Dropout(0.5)
        self.pool = Pooling(pool="mean")
        self.classifier = Classifier(dim=oc, num_classes=num_class)
        
        self.mamba0 = MambaLayers(d_model=ic,n_layers=2)
        self.tb=TransBlock(ic,ws,num_encoder_layers=1)  

        self.ca=StandardCA(ic)
        
        self.mtdc = MTDC(MTDCConv, ic, 3)

        self.lpg = LPG(oc)
        self.c2r = C2R(oc)  # Center-contrast refinement module.

    def forward(self, x):
        x, x1 = x  # Unpack the two input modalities.
        x = torch.cat((x, x1), 1)  # Concatenate along the channel dimension.
        x=self.conv0(x)

        batch_size,c,h,w=x.shape    
        
        x = self.mtdc(x)
        
        x=self.mamba0(x)

        x=self.tb(x)

        x=self.ca(x)
        
        x1= self.inc(x)       
        
        x2=self.down1(x1)
        x2=self.tb1(x2)
        
        x3=self.down2(x2)

        x= self.up1(x3,x2)

        x=self.up2(x,x1)

        x = self.lpg(x)
        x = self.c2r(x)  # Apply center-guided detail refinement.

        x=self.tbU2(x)

        x=self.dropout(x)

        x=x.reshape(batch_size,-1,h*w)
        x=x.permute(0,2,1)
        x=self.pool(x)
        x = self.classifier(x)

        return x

# Backward-compatible alias for existing training scripts and checkpoints.
MyModel = LECMamba


if __name__=='__main__':
    print('start')
    batch_size=32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LECMamba().to(device)

    # inputs (batch_size, ic, height, width)
    dummy_input = torch.randn(batch_size, 64, ws, ws).to(device)
    dummy_input2 = torch.randn(batch_size, 2, ws, ws).to(device)
    dummy=(dummy_input, dummy_input2)

    output = model(dummy)

    print("Output shape:", output.shape)
