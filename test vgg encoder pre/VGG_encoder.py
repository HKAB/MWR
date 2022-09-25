from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import torch

def create_model(arg, model_name):
    ### Create model ###
    if model_name == 'Global_Regressor':
        print('Get Global_Regressor')
        model = Global_Regressor().cuda()
        # model = Global_Regressor()

    if model_name == 'Local_Regressor':
        print('Get Local_Regressor')
        model = Local_Regressor(arg).cuda()

    return model

####################### Regressor Module ######################
class Regressor(nn.Sequential):
    def __init__(self, input_channel, output_channel):
        super(Regressor, self).__init__()
        self.convA = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1)
        self.leakyreluA = nn.ReLU()
        self.convB = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1)
        self.leakyreluB = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.convC = nn.Conv2d(output_channel, 1, kernel_size=1, stride=1)
        self.activation = nn.Tanh()


    def forward(self, x):
        x = self.convA(x)
        x = self.leakyreluA(x)
        x = self.convB(x)
        x = self.leakyreluB(x)
        x = self.dropout(x)
        x = self.convC(x)

        return self.activation(x)
##################################################################

########################## Total Model ###########################

class Global_Regressor(nn.Module):
    def __init__(self):
        super(Global_Regressor, self).__init__()
        self.encoder = ptcv_get_model("bn_vgg16", pretrained=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.regressor = Regressor(1536, 512)


    def forward_siamese(self, x):
        x = self.encoder.features.stage1(x)
        x = self.encoder.features.stage2(x)
        x = self.encoder.features.stage3(x)
        x = self.encoder.features.stage4(x)
        x = self.encoder.features.stage5(x)
        x = self.avg_pool(x)

        return x

    def forward(self, phase, **kwargs):

        if phase == 'train':
            x_1_1, x_1_2, x_2 = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2']
            x_1_1 = self.forward_siamese(x_1_1)
            x_1_2 = self.forward_siamese(x_1_2)
            x_2 = self.forward_siamese(x_2)

            x = torch.cat([x_1_1, x_1_2, x_2], dim=1)

            output = self.regressor(x)

            return output

        elif phase == 'test':
            x_1_1, x_1_2, x_2 = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2']
            x = torch.cat([x_1_1, x_1_2, x_2], dim=1)

            output = self.regressor(x)

            return output

        elif phase == 'extraction':
            x = kwargs['x']
            x = self.forward_siamese(x)

            return x


class VGG_cls_pre(nn.Module):
    def __init__(self):
        super(VGG_cls_pre, self).__init__()
        self.model = create_model(None, "Global_Regressor")
        initial_model = 'utk_coral.pth'
        device = torch.device("cuda:%s" % (0) if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(initial_model, map_location=device)
        model_dict = self.model.state_dict()

        model_dict.update(checkpoint['model_state_dict'])
        self.model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}'".format(initial_model))

        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 40)

    def forward(self, x):
        x = self.model.encoder.features.stage1(x)
        x = self.model.encoder.features.stage2(x)
        x = self.model.encoder.features.stage3(x)
        x = self.model.encoder.features.stage4(x)
        x = self.model.encoder.features.stage5(x)
        x = self.avg_pool(x)
        x = x.view(-1,512)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

