import os 
import torch
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
with torch.no_grad():
    import os
    x = torch.rand((1, 2, 144, 144, 144)).cuda()
    _, model = TransBTS(n_channels=2, num_classes=2, img_dim=144, _conv_repr=True, _pe_type="learned")
    model.cuda()
    y = model(x)
    print(y.shape)
