
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

        # layer_types = [
        #     # in_type, channels, stride
        #     # num layers +1 = num channels
        #     (nn.FieldType(self.gs, [self.build_representation(2)] * 1), 50, 2),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 1), 50, 1),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 1), 50, 1),

        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 1), 100, 2),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 100, 1),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 100, 1),

        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 200,2),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 3), 200,1),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 3), 200,1),
        #     (nn.FieldType(self.gs, [self.build_representation(3)] * 3), None,1),
        # ]
        layer_types = [
            (nn.FieldType(self.gs, [self.build_representation(2)] * 3), 200),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 2), 480),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 6), 480),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 12), 960),
            (nn.FieldType(self.gs, [self.build_representation(3)] * 4), None),
        ]

        blocks = [
            nn.R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=0, stride = 1,bias=False, initialize=False),
            # pool1
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                steerableResBlock(layer_types[i][0], channels = layer_types[i][1], out_type=layer_types[i+1][0], stride=2, features=res_features)
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
        
        self.to_lat = torch.nn.Linear(160*2*2*2*3,16*16*16)
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