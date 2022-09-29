import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels, int(output_channels / 4), 1, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(output_channels / 4))
        self.conv2 = nn.Conv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            stride,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(int(output_channels / 4))
        self.conv3 = nn.Conv2d(
            int(output_channels / 4), output_channels, 1, 1, bias=False
        )
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out



class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(AttentionModule, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)

        out = (1 + out_softmax6) * out_trunk

        return self.last_blocks(out)


class ResidualAttentionModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(ResidualAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 128)
        self.attention_module1 = AttentionModule(128, 128, (56, 56), (28, 28), (14, 14))
        self.residual_block2 = ResidualBlock(128, 256, 2)
        self.attention_module2 = AttentionModule(256, 256, (28, 28), (14, 14), (7, 7))
        self.residual_block3 = ResidualBlock(256, 512, 2)
        self.attention_module3 = AttentionModule(512, 512, (14, 14), (7, 7), (4, 4))
        self.residual_block4 = ResidualBlock(512, 1024, 2)
        self.residual_block5 = ResidualBlock(1024, 1024)
        self.residual_block6 = ResidualBlock(1024, 1024)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('Input shape:', x.shape)
        out = self.conv1(x)
        out = self.mpool1(out)
        # print('input block shape:', out.shape)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        # print('output block1 shape:', out.shape)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        # print('output block2 shape:', out.shape)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        # print('output block3 shape:', out.shape)
        out = self.residual_block4(out)
        # print('output block4 shape:', out.shape)
        out = self.residual_block5(out)
        # print('output block5 shape:', out.shape)
        out = self.residual_block6(out)
        # print('output block6 shape:', out.shape)
        out = self.mpool2(out)

        # print('Embedding shape:', out.shape)

        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def res_attention(in_channels=3, num_classes=1):
    return ResidualAttentionModel(in_channels, num_classes)


if __name__ == '__main__':
    model = res_attention(3,1)
    print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)