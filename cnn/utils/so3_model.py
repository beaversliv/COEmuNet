
import torch
from escnn import group
from escnn import gspaces
from escnn import nn
from typing import Tuple, Union

class steerableResBlock(nn.EquivariantModule):

    def __init__(self, in_type: nn.FieldType, channels: int, out_type: nn.FieldType = None, stride: int = 1, features: str = '3_144'):

        super(steerableResBlock, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        self.gspace = self.in_type.gspace

        if features == 'ico':
            L = 2
            grid = {'type': 'ico'}
        elif features == '2_96':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 4}
        elif features == '2_72':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 3}
        elif features == '3_144':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 6}
        elif features == '3_192':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 8}
        elif features == '3_160':
            L = 3
            grid = {'type': 'thomson', 'N': 160}
        else:
            raise ValueError()

        so3: SO3 = self.in_type.fibergroup

        # number of samples for the discrete Fourier Transform
        S = len(so3.grid(**grid))

        # We try to keep the width of the model approximately constant
        _channels = channels / S
        _channels = int(round(_channels))

        # Build the non-linear layer
        # Internally, this module performs an Inverse FT sampling the `_channels` continuous input features on the `S`
        # samples, apply ELU pointwise and, finally, recover `_channels` output features with discrete FT.
        ftelu = nn.FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        # print(f'ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}')

        self.res_block = nn.SequentialModule(
            nn.R3Conv(in_type, res_type, kernel_size=3, padding=1, bias=False, initialize=False),
            nn.IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            nn.R3Conv(res_type, self.out_type, kernel_size=3, padding=1, stride=stride, bias=False, initialize=False),
        )

        if stride > 1:
            self.downsample = nn.PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = nn.R3Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: nn.GeometricTensor):

        assert input.type == self.in_type
        output = self.skip(self.downsample(input)) + self.res_block(input) 
        # print(output.shape)
        return output

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape


class steerablEncoder(torch.nn.Module):
    def __init__(self,res_features: str = '2_96', init: str = 'delta'):

        super(steerablEncoder, self).__init__()
        # the model is equivariant under all planar rotations
        self.gs = gspaces.rot3dOnR3()

        # the group SO(3)
        self.G: SO3 = self.gs.fibergroup
        # store the input type for wrapping the images into a geometric tensor during the forward pass
        self.in_type = nn.FieldType(self.gs, [self.gs.trivial_repr])

        self._init = init

        layer_types = [
            # in_type, channels, stride
            # num layers +1 = num channels
            (nn.FieldType(self.gs, [self.build_representation(2)] * 2), 200, 2),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 200, 1),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 200, 1),

            (nn.FieldType(self.gs, [self.build_representation(3)] * 3), 480, 2),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 3), 480, 1),
            # (nn.FieldType(self.gs, [self.build_representation(3)] * 3), 480, 1),

            (nn.FieldType(self.gs, [self.build_representation(3)] * 4), 960,2),
            # (nn.FieldType(self.gs, [self.build_representation(3)] * 4), 960,1),
            # (nn.FieldType(self.gs, [self.build_representation(3)] * 3), 960,1),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 4), None,1),
        ]
        # layer_types = [
        #     (nn.FieldType(self.gs, [self.build_representation(2)] * 3), 200),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 480),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 6), 480),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 12), 960),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 4), None),
        # ]

        blocks = [
            nn.R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=0, stride = 1,bias=False, initialize=False),
            # pool1
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                steerableResBlock(layer_types[i][0], channels = layer_types[i][1], out_type=layer_types[i+1][0], stride=layer_types[i+1][2], features=res_features)
            )
        blocks.append(
                      nn.PointwiseAvgPoolAntialiased3D(in_type=layer_types[-1][0], sigma=.33, stride=2, padding=1))
        
        self.blocks = nn.SequentialModule(*blocks)


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.R3Conv):
                if self._init == 'he':
                    nn.init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    nn.init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return group.directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):

        input = nn.GeometricTensor(input, self.in_type)

        features = self.blocks(input)

        return features

class steerableDecoder(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(steerableDecoder, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Example: Halving the channels and doubling the spatial dimension with each step
        for i in range(3):
            self.layers.append(torch.nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, stride=1,padding=1))
            self.layers.append(torch.nn.BatchNorm2d(int(in_channels / 2)))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))  # Upsampling
            in_channels = int(in_channels/2)

        # Final convolution to get the desired number of output channels (1 in this case)
        self.layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class steerabelNet(torch.nn.Module):
    def __init__(self):
        super(steerabelNet, self).__init__()
        self.encoder0 = steerablEncoder(res_features = '2_96', init = 'delta')
        self.encoder1 = steerablEncoder(res_features = '2_96', init = 'delta')
        self.encoder2 = steerablEncoder(res_features = '2_96', init = 'delta')
        
        self.to_lat = torch.nn.Linear(160*8*8*8*3,16*16*16)
        self.to_dec = torch.nn.Linear(16*16*16,64*8*8)
        self.decoder= steerableDecoder(in_channels=64, out_channels=1)
        
        
    def forward(self, x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])

        # unwrap the output GeometricTensor
        x0 = x0.tensor
        x1 = x1.tensor
        x2 = x2.tensor
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1) 
        # x shape (batch size, 32*4*4*4*3)
        x = torch.cat([x0, x1, x2], dim = -1)
 
        # (batch, 16*16*16)
        x_latent = self.to_lat(x) #dense layer
        x = torch.nn.ReLU()(self.to_dec(x_latent)) # latent space
        x = x.view(-1, 64, 8, 8)
      
	# shape (batch_size,64,8,8)
        output = self.decoder(x)
        return x_latent, output

class ClsSO3VoxConvModel(torch.nn.Module):
    def __init__(self,freq:int,scale:int):
        super(ClsSO3VoxConvModel, self).__init__()
        self.freq = freq
        self.scale = scale

        self.r3_act = gspaces.rot3dOnR3(self.freq)
        g = self.r3_act.fibergroup
        reg_repr = g.bl_regular_representation(self.freq)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r3_act, [self.r3_act.trivial_repr])
        self.input_type = in_type
        # convolution 0 
        out_type = nn.FieldType(self.r3_act, 5*self.scale*[self.r3_act.irrep(0)] + 3*self.scale*[self.r3_act.irrep(1)] + 5*self.scale*[self.r3_act.irrep(2)] )

        self.conv0 = nn.R3Conv(in_type, out_type, kernel_size=5, padding=2)
        # print(f'conv 0: {in_type.size} {out_type.size}')
        
        in_type = out_type
        if self.freq == 2:
            activation1 = nn.FourierELU(self.r3_act, 7*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation1 = nn.FourierELU(self.r3_act, 3*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)
        out_type = activation1.in_type
        if self.freq == 2:
            next_type = nn.FieldType(self.r3_act, 10*self.scale*[self.r3_act.irrep(0)] + 6*self.scale*[self.r3_act.irrep(1)] + 10*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = nn.FieldType(self.r3_act, [reg_repr])

        self.block1 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, 3, 1),
            nn.IIDBatchNorm3d(out_type),
            activation1,
            nn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 39-240-78
        # self.skip1 = nn.SequentialModule(
        #     nn.R3Conv(in_type, next_type, 1),
        #     nn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        # )
        # print(f'block 1: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        if self.freq == 2:
            activation2 = nn.FourierELU(self.r3_act, 14*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation2 = nn.FourierELU(self.r3_act, 6*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)

        out_type = activation2.in_type
        if self.freq == 2:
            next_type = nn.FieldType(self.r3_act, 26*self.scale*[self.r3_act.irrep(0)] + 28*self.scale*[self.r3_act.irrep(1)] + 26*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = nn.FieldType(self.r3_act, 3*[reg_repr])

        self.block2 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, 3, 1),
            nn.IIDBatchNorm3d(out_type),
            activation2,
            nn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 78-480-240
        # self.skip2 = nn.SequentialModule(
        #     nn.R3Conv(in_type, next_type, 1),
        #     nn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        # )
        # print(f'block 2: {in_type.size} {out_type.size} {next_type.size}')
        
        in_type = next_type
        if self.freq == 2:
            activation3 = nn.FourierELU(self.r3_act, 14*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation3 = nn.FourierELU(self.r3_act, 6*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)

        out_type = activation3.in_type
        if self.freq == 2:
            next_type = nn.FieldType(self.r3_act, 53*self.scale*[self.r3_act.irrep(0)] + 54*self.scale*[self.r3_act.irrep(1)] + 53*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = nn.FieldType(self.r3_act, 6*[reg_repr])

        self.block3 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, 3, 1),
            nn.IIDBatchNorm3d(out_type),
            activation3,
            nn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 240-480-480
        # self.skip3 = nn.SequentialModule(
        #     nn.R3Conv(in_type, next_type, 1),
        #     nn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        # )
        # print(f'block 3: {in_type.size} {out_type.size} {next_type.size}')
        
        in_type = next_type
        if self.freq == 2:
            activation4 = nn.FourierELU(self.r3_act, 27*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation4 = nn.FourierELU(self.r3_act, 11*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)

        out_type = activation4.in_type
        if self.freq == 2:
            next_type = nn.FieldType(self.r3_act, 35*self.scale*[self.r3_act.irrep(0)] + 34*self.scale*[self.r3_act.irrep(1)] + 35*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = nn.FieldType(self.r3_act, 4*[reg_repr])

        self.block4 = nn.SequentialModule(
            nn.R3Conv(in_type, out_type, 3, 1),
            nn.IIDBatchNorm3d(out_type),
            activation4,
            nn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 480-960-312
        # self.skip4 = nn.SequentialModule(
        #     nn.R3Conv(in_type, next_type, 1),
        #     nn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        # )
        # print(f'block 4: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        out_type = nn.FieldType(self.r3_act, [reg_repr] * 128)
        self.conv5 = nn.R3Conv(in_type, out_type, 3)

        # self.inv_layer = nn.NormPool(out_type)
        # next_type = self.inv_layer.out_type
        # print(f'inv: {in_type.size} {out_type.size} {next_type.size}')

        # self.fc1 = torch.nn.Sequential(
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ELU(),
        # )
        # self.fc2 = torch.nn.Sequential(
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.BatchNorm1d(128),
        #     torch.nn.ELU(),
        # )
        # self.fc3 = torch.nn.Linear(128, 40)

    def forward(self, input):
        bdim = input.shape[0]
        x = self.input_type(input)
        x = self.conv0(x)
        # print(f"conv0 {x.shape}")

        x = self.block1(x)
        # x2 = self.skip1(x)
        # print(f"block1 {x.shape}")

        x = self.block2(x)
        # x2 = self.skip2(x)
        # print(f"block3 {x.shape}")
        # x = x1 + x2
        
        x = self.block3(x)
        # x2 = self.skip3(x)
        # print(f"block3 {x.shape}")
        # x = x1 + x2

        x = self.block4(x)
        # x2 = self.skip4(x)
        # print(f"block4 {x.shape}")
        # x = x1 + x2
        x = self.conv5(x)
        # print(f"conv5 {x.shape}")

        # x = self.inv_layer(x)
        # # print(f"inv_layer {x.shape}")
        
        # x = x.tensor.reshape(bdim, -1)
        # x_feat = x
        # # print(f"torch tensor {x.shape}")
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        return x

class ClsSO3Net(torch.nn.Module):
    def __init__(self):
        super(ClsSO3Net,self).__init__()
        self.encoder0 = ClsSO3VoxConvModel(freq=1,scale=1)
        self.encoder1 = ClsSO3VoxConvModel(freq=1,scale=1)
        self.encoder2 = ClsSO3VoxConvModel(freq=1,scale=1)

        self.to_lat = torch.nn.Linear(1280*2*2*2*3,16*16*16)
        self.to_dec = torch.nn.Linear(16*16*16,64*8*8)
        self.decoder= steerableDecoder(in_channels=64, out_channels=1)
    
    def forward(self, x):
        x0 = self.encoder0(x[:, 0:1, :, :, :])
        x1 = self.encoder1(x[:, 1:2, :, :, :])
        x2 = self.encoder2(x[:, 2:3, :, :, :])

        # unwrap the output GeometricTensor
        x0 = x0.tensor
        x1 = x1.tensor
        x2 = x2.tensor
        x0 = torch.flatten(x0, start_dim=1)   
        x1 = torch.flatten(x1, start_dim=1)   
        x2 = torch.flatten(x2, start_dim=1) 
        # x shape (batch size, 32*4*4*4*3)
        x = torch.cat([x0, x1, x2], dim = -1)
 
        # (batch, 16*16*16)
        x_latent = self.to_lat(x) #dense layer
        x = torch.nn.ReLU()(self.to_dec(x_latent)) # latent space
        x = x.view(-1, 64, 8, 8)
      
	# shape (batch_size,64,8,8)
        output = self.decoder(x)
        return x_latent, output