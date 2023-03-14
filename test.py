import os 
import torch
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models.HDenseFormer import HDenseFormer_32
from models.HDenseFormer_2D import HDenseFormer_2D_32
from models.Hecktor20Top1.model import hecktertop1
from models.UNETR import UNETR

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
with torch.no_grad():
    import os
    # x = torch.rand((1, 2, 144, 144, 144)).cuda()
    x = torch.rand((1, 2, 384, 384)).cuda()
    # _, model = TransBTS(n_channels=2, num_classes=2, img_dim=144, _conv_repr=True, _pe_type="learned")
    # model = HDenseFormer_32(in_channels=2,
    #                         n_cls=2,
    #                         image_size=(144,144,144),
    #                         transformer_depth=16)
    # model = hecktertop1(in_channels=2,n_cls=2)

    model = HDenseFormer_2D_32(in_channels=2,
                            n_cls=2,
                            image_size=(384,384),
                            transformer_depth=16)
    # model = UNETR(in_channels=2,
    #                 out_channels=2,
    #                 img_size=(144,144,144),
    #                 feature_size=16,
    #                 hidden_size=768,
    #                 mlp_dim=3072,
    #                 num_heads=12,
    #                 pos_embed='perceptron',
    #                 norm_name='instance',
    #                 conv_block=True,
    #                 res_block=True,
    #                 dropout_rate=0.0)

    model.cuda()
    y = model(x)
    print(y[0].shape)
