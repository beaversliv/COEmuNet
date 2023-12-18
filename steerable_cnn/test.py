
from typing import Tuple, Union
from escnn import group,gspaces
import escnn.nn as ecnn 
import torch
from torch import nn
import numpy as np

class ResBlock(ecnn.EquivariantModule):

    def __init__(self, in_type: ecnn.FieldType, channels: int, out_type: ecnn.FieldType = None, stride: int = 1, features: str = '2_96'):

        super(ResBlock, self).__init__()

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
        ftelu = ecnn.FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        # print(f'ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}')

        self.res_block = ecnn.SequentialModule(
            ecnn.R3Conv(in_type, res_type, kernel_size=3, padding=1, bias=False, initialize=False),
            ecnn.IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            ecnn.R3Conv(res_type, self.out_type, kernel_size=3, padding=1, stride=stride, bias=False, initialize=False),
        )

        if stride > 1:
            self.downsample = ecnn.PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = ecnn.R3Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: ecnn.GeometricTensor):

        assert input.type == self.in_type
        return self.skip(self.downsample(input)) + self.res_block(input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape

class encoder(nn.Module):
    def __init__(self, pool: str = "snub_cube", res_features: str = '2_96', init: str = 'delta'):

        super(encoder, self).__init__()

        self.gs = gspaces.rot3dOnR3()

        self.in_type = ecnn.FieldType(self.gs, [self.gs.trivial_repr])

        self._init = init
        # For pooling, we map the features to a spherical representation (bandlimited to freq 2)
        # Then, we apply pointwise ELU over a number of samples on the sphere and, finally, compute the average
        # # (i.e. recover only the frequency 0 component of the output features)
        if pool == "icosidodecahedron":
            # samples the 30 points of the icosidodecahedron
            # this is only perfectly equivarint to the 12 tethrahedron symmetries
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif pool == "snub_cube":
            # samples the 24 points of the snub cube
            # this is perfectly equivariant to all 24 rotational symmetries of the cube
            grid = self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
        else:
            raise ValueError(f"Pooling method {pool} not recognized")

        # pool1 = QuotientFourierELU(self.gs, (False, -1), 4, irreps=self.gs.fibergroup.bl_irreps(2), out_irreps=self.gs.fibergroup.bl_irreps(0), grid=grid)
        ftgpool = ecnn.QuotientFourierELU(self.gs, (False, -1), 32, irreps=self.gs.fibergroup.bl_irreps(2), out_irreps=self.gs.fibergroup.bl_irreps(0), grid=grid)

        layer_types = [
            # in_type, channels, stride
            # r3conv
            (ecnn.FieldType(self.gs, [self.build_representation(2)] * 4), 200, 1),
            # b1
            (ecnn.FieldType(self.gs, [self.build_representation(3)] * 8), 200, 2),
            # b2
            (ecnn.FieldType(self.gs, [self.build_representation(3)] * 8), 480, 1),
            # (ecnn.FieldType(self.gs, [self.build_representation(3)] * 8), 480, 1),

            (ecnn.FieldType(self.gs, [self.build_representation(3)] * 16), 960,2),
            (ecnn.FieldType(self.gs, [self.build_representation(3)] * 16), 960,1),
            # (ecnn.FieldType(self.gs, [self.build_representation(3)] * 16), 960,1),

            (ecnn.FieldType(self.gs, [self.build_representation(3)] * 32), 1920,2),
            # (ecnn.FieldType(self.gs, [self.build_representation(3)] * 32), 1920,1),
            (ecnn.FieldType(self.gs, [self.build_representation(3)] * 32), None,1),
        ]

        blocks = [
            ecnn.R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, bias=False, initialize=False),
            # pool1
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], channels = layer_types[i][1], out_type=layer_types[i+1][0], stride=layer_types[i][2], features=res_features)
            )

        final_features = ftgpool.in_type
        blocks += [
            ecnn.R3Conv(layer_types[-1][0], final_features, kernel_size=3, padding=0, bias=False, initialize=False),
            ftgpool,
        ]
        C = ftgpool.out_type.size
        self.blocks = ecnn.SequentialModule(*blocks)


    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, ecnn.R3Conv):
                if self._init == 'he':
                    ecnn.init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    ecnn.init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
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

        input = ecnn.GeometricTensor(input, self.in_type)

        features = self.blocks(input)

        return features



gs = gspaces.rot3dOnR3()



in_type  = ecnn.FieldType(gs, [gs.trivial_repr])
b1      = ecnn.FieldType(gs, [gs.trivial_repr])
b2      = ecnn.FieldType(gs, 4*[gs.trivial_repr])
b3      = ecnn.FieldType(gs, 8*[gs.trivial_repr])
b4      = ecnn.FieldType(gs, 16*[gs.trivial_repr])
b5      = ecnn.FieldType(gs, 32*[gs.trivial_repr])

# net = ecnn.SequentialModule(
#     ecnn.R3Conv(in_type, b1, kernel_size=3, padding=0, stride=1,bias=False, initialize=False),
#     ResBlock(b1, channels = 64, out_type=b2, stride=2, features='2_72'),
#     ResBlock(b2, channels = 64, out_type=b3, stride=1, features='2_72'),
#     ResBlock(b3, channels = 64, out_type=b3, stride=1, features='2_72'),
#     ResBlock(b3, channels = 128, out_type=b4, stride=2, features='2_72'),
#     ResBlock(b4, channels = 128, out_type=b4, stride=1, features='2_72'),
#     ResBlock(b4, channels = 128, out_type=b4, stride=1, features='2_72'),
#     ResBlock(b4, channels = 256, out_type=b5, stride=2, features='2_72'),
#     ResBlock(b5, channels = 256, out_type=b5, stride=1, features='2_72'),
#     ResBlock(b5, channels = 256, out_type=b5, stride=1, features='2_72')    
# )
net = ResBlock(b1, channels = 64, out_type=b2, stride=2, features='2_72')
net.eval()
net_exported = net.export()
print(net)
print(net_exported)

# check that the two models are equivalent
x0 = torch.randn(5, 1, 64, 64, 64)
x0 = ecnn.GeometricTensor(x0,b1)
y0 = net(x0).tensor
print(y0.shape)
y1 = net_exported(x0.tensor)
assert torch.allclose(y0,y1) # True

# block1 = ResBlock(b1, channels = 64, out_type=b2, stride=2, features='2_72')
# block2 = ResBlock(b2, channels = 64, out_type=b3, stride=1, features='2_72')
# block3 = ResBlock(b3, channels = 64, out_type=b3, stride=1, features='2_72')

# block4 = ResBlock(b3, channels = 128, out_type=b4, stride=2, features='2_72')
# block5 = ResBlock(b4, channels = 128, out_type=b4, stride=1, features='2_72')
# block6 = ResBlock(b4, channels = 128, out_type=b4, stride=1, features='2_72')

# block7 = ResBlock(b4, channels = 256, out_type=b5, stride=2, features='2_72')
# block8 = ResBlock(b5, channels = 256, out_type=b5, stride=1, features='2_72')
# block9 = ResBlock(b5, channels = 256, out_type=b5, stride=1, features='2_72')

# grid = gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
# ftgpool = QuotientFourierELU(gs, (False, -1), 32, irreps=gs.fibergroup.bl_irreps(2), out_irreps=gs.fibergroup.bl_irreps(0), grid=grid)

# final_features = ftgpool.in_type
# conv1 = R3Conv(b5, final_features, kernel_size=3, padding=0, bias=False, initialize=False)
# ftgpool

# encoder0 = encoder(pool='snub_cube', res_features='2_72', init='he')
# encoder0.init()
# encoder0.eval()
# net_exported = encoder0.export()
# print(encoder0)
# print(net_exported)
# x0 = torch.randn(5, 1, 64, 64, 64)
# y = encoder0(x0)
# input = GeometricTensor(x0, in_type)

# x = conv0(input)
# x = block1(x)
# x = block2(x)
# x = block3(x)
# x = block4(x)
# x = block5(x)
# x = block6(x)
# x = block7(x)
# x = block8(x)
# x = block9(x)
# x = conv1(x)
# y = ftgpool(x)
# y = y.tensor
# print(type(y))
# print(y.shape)

