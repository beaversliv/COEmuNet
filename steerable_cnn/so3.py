import torch
from so3_model       import steerabelNet,steerablEncoder,ClsSO3Net

if __name__ == '__main__':
    # model = steerabelNet()
    # model = steerablEncoder(res_features = '2_96', init = 'delta')
    model  = ClsSO3Net()
    a = torch.randn(5, 3, 64, 64, 64)

    y0 = model(a)
    print('final',y0.shape)