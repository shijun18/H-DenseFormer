import torch

def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print(params)
    print(1e6)
    print('%.3f GFLOPs' %(macs/1e9))
    print('%.3f M' % (params/1e6))