# install the following packages pytorchcv, mtcnn, imgaug

from mtcnn import MTCNN
import cv2
import os
from PIL import Image
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import numpy as np
from imgaug import augmenters as iaa

# load model
def create_model(arg, model_name):
    ### Create model ###
    if model_name == 'Global_Regressor':
        print('Get Global_Regressor')
        model = Global_Regressor()
        # model = Global_Regressor()
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
        # initial_model = 'utk_coral.pth'
        # device = torch.device("cuda:%s" % (0) if torch.cuda.is_available() else "cpu")
        # checkpoint = torch.load(initial_model, map_location=device)
        # model_dict = self.model.state_dict()

        # model_dict.update(checkpoint['model_state_dict'])
        # self.model.load_state_dict(model_dict)
        # print("=> loaded checkpoint '{}'".format(initial_model))

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

def load_model(checkpoint_path):
    model = VGG_cls_pre()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    print('Load model successfully!')
    return model

# Face detect:
detector = MTCNN()
def face_detect(file_path, thickness = 2, detector = detector, color = (255, 0, 0)):
  try:
    img = cv2.imread(file_path)
    if len(img.shape)==2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face by MTCNN
    results = detector.detect_faces(img)
    if results == []:
      print('No face detect!')
      return None, None, None
    else:
      x,y,w,h = results[0]['box']
      x  = int(x - 0.25*w)
      y = int(y - 0.25*h)
      if x<0:
        x = 0
      if y < 0:
        y = 0
      x_end = int(x + 1.5*w)
      y_end = int(y + 1.5*h)
      if x>img.shape[0]:
        x = int(img.shape[0])
      if y > img.shape[1]:
        y = int(img.shape[1])

      image = img[y: y_end, x:x_end]
      
      image_viz = cv2.rectangle(img, (x,y), (x_end, y_end), color, thickness)
      # plt.imshow(image_viz)
      
      print('Face detect succesfully!')
      return image, image_viz, (x, y, x_end-x, y_end-y)
  except:
    print('Cant read file:', file_path, 'please check file!')
    return None, None, None

# Utils functions:
imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def ImgAugTransform_Test(img):

    aug = iaa.Sequential([
            iaa.CropToFixedSize(width=224, height=224, position="center")
        ])

    img = np.array(img)
    img = aug(image=img)
    return img

def apparent_age(m, predict, age_range):
    predict = m(predict)
    return torch.sum(predict * age_range, dim=1)

def age_estimate(model, img_path, face_detect = face_detect, save_folder='result', img_size=224, imagenet_stats = imagenet_stats, m = torch.nn.Softmax(dim=1), age_range=torch.arange(21, 61).float()):
    face_img, image_viz, box = face_detect(img_path)
    if face_img is not None:
        face_img = Image.fromarray(face_img)
        img = face_img.resize((img_size, img_size))
        img = ImgAugTransform_Test(img).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        dtype = img.dtype
        mean = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=img.device)
        std = torch.as_tensor(imagenet_stats['mean'], dtype=dtype, device=img.device)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])
            
        img = img[None,:]
        predict = model(img)
        age_predict = apparent_age(m, predict, age_range)
        age_predict = int(age_predict.detach().numpy())
        image_viz = cv2.putText(image_viz,str(age_predict), (box[0]+int(box[2]/2)-30,box[1]+int(box[3])-50),cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 4, cv2.LINE_AA)
        save_path = os.path.join(save_folder, img_path)
        image_viz = cv2.cvtColor(image_viz, cv2.COLOR_RGB2BGR)
        print(str(age_predict))
        cv2.imwrite(save_path, image_viz)
        # plt.imshow(image_viz)
    else:
        print('Cant estimate age, an error occurs!')

if __name__=="__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # Load model
    model = load_model('vgg_epoch_10.pth')

    img_path = 'yoona3.jpg'

    age_estimate(model, img_path)

