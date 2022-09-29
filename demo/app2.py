from flask import Flask, render_template, redirect, url_for, request, jsonify
app = Flask(__name__, template_folder='templates')
import cv2 as cv2
import numpy as np
import urllib.request
from PIL import Image
import io
import random

from mtcnn import MTCNN
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import numpy as np
from imgaug import augmenters as iaa
import math
from tqdm import tqdm

device = torch.device("cuda:%s" % (0) if torch.cuda.is_available() else "cpu")

@app.route('/')
def home():
   return render_template('home.html')

# load model
def create_model(arg, model_name):
    ### Create model ###
    if model_name == 'Global_Regressor':
        print('Get Global_Regressor')
        model = Global_Regressor()
        # model = Global_Regressor()
    else:
        model = Local_Regressor()
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
class Local_Regressor(nn.Module):
    def __init__(self, reg_num=5):
        super(Local_Regressor, self).__init__()

        self.reg_num = reg_num
        self.encoder = nn.ModuleList([ptcv_get_model("bn_vgg16", pretrained=True) for _ in range(self.reg_num)])
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.regressor = nn.ModuleList([Regressor(1536, 512) for _ in range(self.reg_num)])

    def forward_siamese(self, x, idx):
        x = self.encoder[idx].features(x)
        x = self.avg_pool(x)

        return x

    def forward(self, phase, **kwargs):

        if phase == 'train':
            x_1, x_2, x_test, idx = kwargs['x_1'], kwargs['x_2'], kwargs['x_test'], kwargs['idx']
            x_1, x_2, x_test = self.forward_siamese(x_1, idx), self.forward_siamese(x_2, idx), self.forward_siamese(x_test, idx)

            x_cat = torch.cat([x_1, x_2, x_test], dim=1)
            outputs = self.regressor[idx](x_cat)

            return outputs.squeeze()

        elif phase == 'test':
            x_1_1, x_1_2, x_2, idx = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2'], kwargs['idx']
            x = torch.cat([x_1_1, x_1_2, x_2], dim=1)
            outputs = self.regressor[idx](x)

            return outputs

        elif phase == 'extraction':
            x, idx = kwargs['x'], kwargs['idx']
            x = self.forward_siamese(x, idx)
            return x

##################################################################

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


def load_mwr_model(pth_path, dataset='utk', experiment_setting='coral'):
    model = Local_Regressor()
    model.to(device)

    ### Load network parameters ###
#     initial_model = os.path.join(pth_path, '%s_%s_local.pth'%(str(dataset).upper(), experiment_setting.lower()))

#     checkpoint = torch.load(initial_model, map_location=device)
#     model_dict = model.state_dict()

#     model_dict.update(checkpoint['model_state_dict'])
#     model.load_state_dict(model_dict)
#     print("=> loaded checkpoint '{}'".format(initial_model))

    model.eval()
    
    return model


# Face detect:
detector = MTCNN()
def face_detect(img, thickness = 2, detector = detector, color = (255, 0, 0)):
  try:
    # img = cv2.imread(img)
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
    print('Cant read file')
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
        # cv2.imwrite(save_path, image_viz)
        # plt.imshow(image_viz)
        return image_viz
    else:
        print('Cant estimate age, an error occurs!')
        return None

def get_age_bounds(age, age_unique, tau=0.1, mode='geometric'):
    lb_list = []
    up_list = []

    if mode == 'geometric':

        for age_tmp in range(age + 1):

            lb_age = sum(np.arange(age) < age_tmp * math.exp(-tau))
            up_age = sum(np.arange(age) < age_tmp * math.exp(tau))

            lb_sub_unique = abs(age_unique - lb_age)
            up_sub_unique = abs(age_unique - up_age)

            lb_nearest = np.argsort(lb_sub_unique)
            up_nearest = np.argsort(up_sub_unique)

            lb_list.append(int(age_unique[lb_nearest[0]]))
            up_list.append(int(age_unique[up_nearest[0]]))

    elif mode == 'arithmetic':

        for age_tmp in range(age + 1):

            if age_tmp in age_unique:
                lb_age = sum(np.arange(age) < age_tmp - tau)
                up_age = sum(np.arange(age) < age_tmp + tau)

                lb_sub_unique = abs(age_unique - lb_age)
                up_sub_unique = abs(age_unique - up_age)

                lb_nearest = np.argsort(lb_sub_unique)
                up_nearest = np.argsort(up_sub_unique)

                lb_list.append(age_unique[lb_nearest[0]])
                up_list.append(age_unique[up_nearest[0]])

            else:
                lb_list.append(sum(np.arange(age) < age_tmp - tau))
                up_list.append(sum(np.arange(age) < age_tmp + tau))

    return lb_list, up_list
    
def initial_prediction(train_data, test_data, features, n_neighbors):

    f_tr, f_te = torch.as_tensor(features['train']), torch.as_tensor(features['test'])
    print("f_tr, f_te: ", f_tr.shape, f_te.shape)
    tr_age = torch.as_tensor(train_data['age'])

    cdist = torch.cdist(f_te, f_tr)
    distances, indices = torch.topk(cdist, n_neighbors, dim=1, largest=False)

    indices = indices.cpu().detach().numpy()
    init_pred = (tr_age[indices].float().mean(dim=1) + 0.5).int()

    return init_pred.cpu().detach().numpy()

def get_random_pairs_local_regression(train_data, reg_bound, tau=6):

    data_age_max = int(train_data['age'].max())
    data_age = train_data['age'].to_numpy().astype('int')
    lb_list, up_list = get_age_bounds(int(data_age_max), np.unique(data_age), tau)

    loss_total = [[] for _ in range(len(reg_bound))]
    index_y2_total = [[] for _ in range(len(reg_bound))]

    for i, age_y1 in enumerate(tqdm(data_age)):
        for r in range(len(reg_bound)):
            if (age_y1 >= lb_list[reg_bound[r][0]]) and (age_y1 <= reg_bound[r][1]):
                # age_y1, age_y2 are lb and ub ages

                age_y2 = up_list[age_y1]
                index_y2 = np.where(data_age == age_y2)[0]
                index_y2_total[r].append(index_y2)
                # index_test: get all ages in between

            else:
                loss_total[r].append([])
                index_y2_total[r].append([])

    return index_y2_total


def select_random_reference_local_regression(train_data, pair_idx, reg_bound, limit=5, tau=5):
    # ref vector
    refer_idx, refer_pair_idx = [[] for _ in range(len(reg_bound))], [[] for _ in range(len(reg_bound))]

    data_age = np.array(train_data['age']).astype('int')
    data_age_max = data_age.max()
    lb_list, up_list = get_age_bounds(int(data_age_max), np.unique(data_age), tau)

    for i in range(90):
        for r in range(len(reg_bound)):
            if (i >= lb_list[reg_bound[r][0]]) and (i <= reg_bound[r][1]):

                ref_image = train_data.loc[np.array(train_data['age']).astype('int') == i]
                idx = train_data.loc[np.array(train_data['age']).astype('int') == i].index.to_numpy()

                if len(ref_image) > 0:
                    y_1_idx_tmp = np.random.choice(idx, 1)
                    y_2_idx_tmp = np.random.choice(np.array(pair_idx[r])[y_1_idx_tmp][0], 1)
                    # np.array(pair_idx[r])[y_1_idx_tmp][0] here because np.array(pair_idx[r])[y_1_idx_tmp]
                    # return a list of array
                    refer_idx[r].append(y_1_idx_tmp.tolist())
                    refer_pair_idx[r].append(y_2_idx_tmp.tolist())

                else:
                    refer_idx[r].append([])
                    refer_pair_idx[r].append([])

            else:
                refer_idx[r].append([])
                refer_pair_idx[r].append([])

    return refer_idx, refer_pair_idx

def MWR_local_regression(train_data, test_data, features, refer_idx, refer_idx_pair, global_prediction, reg_bound, model, device, tau=0.2, reg_num=5):
    pred_age_final = []
    train_data_age_unique = np.unique(train_data['age'])
    train_data_age_total = train_data['age'].to_numpy().reshape(-1)

    f_tr = torch.as_tensor(features['train']).to(device)
    f_te = torch.as_tensor(features['test']).to(device)

    lb_list_global, up_list_global = get_age_bounds(int(train_data['age'].max()), np.unique(train_data['age']), tau)
    lb_list_total, up_list_total = [], []
    lb_list_half_total, up_list_half_total = [], []

    for reg_num_tmp in range(reg_num):
        max_age = up_list_global[reg_bound[reg_num_tmp][1]]
        min_age = lb_list_global[reg_bound[reg_num_tmp][0]]
        train_age_list = np.where((train_data_age_total >= min_age) & (train_data_age_total <= max_age))[0]
        train_age_list = train_data_age_total[train_age_list]
        lb_list, up_list = get_age_bounds(int(train_age_list.max()), np.unique(train_age_list), tau)
        lb_list_half, up_list_half = get_age_bounds(int(train_age_list.max()), np.unique(train_age_list), tau / 2)

        lb_list_total.append(lb_list), up_list_total.append(up_list)
        lb_list_half_total.append(lb_list_half), up_list_half_total.append(up_list_half)

    with torch.no_grad():

        max_local_iter = 10
        memory = np.zeros(shape=(len(test_data), max_local_iter))

        lb_final_total, up_final_total, reg_idx = [], [], []

        for i in tqdm(range(0, len(test_data))):

            reg_num_list = []
            refine_list = []
            iteration = 0

            age = int(global_prediction[i])
            for tmp in range(reg_num):
                if age in np.arange(reg_bound[tmp][0], reg_bound[tmp][1] + 1):
                    reg_num_list.append(tmp)

            while True:
                for tmp_idx, reg_num_idx in enumerate(reg_num_list):

                    lb_age = int(np.argsort(np.abs(np.log(age) - np.log(train_data_age_unique) - tau / 2))[0])
                    lb_age = int(train_data_age_unique[lb_age])
                    up_age = int(up_list_total[reg_num_idx][lb_age])

                    if tmp_idx == 0:
                        lb_final = lb_age
                        up_final = up_age
                        reg_idx_final = reg_num_idx

                    lb_age, up_age = np.array([lb_age]), np.array([up_age])

                    invalid_lb = np.setdiff1d(lb_age, (train_data_age_unique).astype('int'), True)
                    invalid_up = np.setdiff1d(up_age, (train_data_age_unique).astype('int'), True)

                    if len(invalid_lb) != 0 or len(invalid_up) != 0:
                        invalid_idx_lb, invalid_idx_up = [], []

                        for invalid_age in invalid_lb:
                            invalid_idx_lb.append(list(np.where(invalid_age == lb_age)[0]))
                        for invalid_age in invalid_up:
                            invalid_idx_up.append(list(np.where(invalid_age == up_age)[0]))

                        invalid_idx = sum(invalid_idx_lb, []) + sum(invalid_idx_up, [])

                        lb_age = np.delete(lb_age, invalid_idx)
                        up_age = np.delete(up_age, invalid_idx)

                    idx_1_final = [refer_idx[reg_num_idx][lb_age[tmp]] for tmp in range(len(lb_age))]
                    idx_2_final = [refer_idx_pair[reg_num_idx][lb_age[tmp]] for tmp in range(len(up_age))]

                    idx_1_final = np.array(sum(idx_1_final, []))
                    idx_2_final = np.array(sum(idx_2_final, []))

                    if len(idx_1_final) == 0:
                        idx_1_final = [refer_idx[reg_num_idx][lb_age[tmp] - 1] for tmp in range(len(lb_age))]
                        lb_age = lb_age - 1
                    if len(idx_2_final) == 0:
                        idx_2_final = [refer_idx_pair[reg_num_idx][lb_age[tmp] + 1] for tmp in range(len(up_age))]
                        up_age = up_age + 1

                    idx_1_final = np.array(idx_1_final).reshape(-1)
                    idx_2_final = np.array(idx_2_final).reshape(-1)

                    test_duplicate = [i] * len(idx_1_final)

                    feature_1, feature_2, feature_test = f_tr[reg_num_idx][idx_1_final].reshape(-1, 512, 1, 1), f_tr[reg_num_idx][idx_2_final].reshape(-1, 512, 1, 1), \
                                                         f_te[reg_num_idx][test_duplicate].reshape(-1, 512, 1, 1)

                    outputs = model('test', x_1_1=feature_1, x_1_2=feature_2, x_2=feature_test, idx=reg_num_idx)
                    # print("feature_1", feature_1.shape)
                    # print("feature_2", feature_2.shape)
                    # print("feature_test", feature_test.shape)
                    outputs = outputs.squeeze().cpu().detach().numpy().reshape(-1)

                    up_age = np.array([up_age[tmp].repeat(len(refer_idx_pair[reg_num_idx][lb_age[tmp]])) for tmp in range(len(up_age))]).reshape(-1)
                    lb_age = np.array([lb_age[tmp].repeat(len(refer_idx[reg_num_idx][lb_age[tmp]])) for tmp in range(len(lb_age))]).reshape(-1)

                    mean = (np.log(up_age) + np.log(lb_age)) / 2
                    tau = abs(mean - np.log(lb_age))

                    refined_age_tmp = np.mean([np.exp(outputs[k] * (tau[k]) + mean[k]) for k in range(len(outputs))])
                    # print("refined_age_tmp", refined_age_tmp)
                    if sum(outputs == 1) == len(outputs):
                        refined_age_tmp = up_age.max()
                    if sum(outputs == -1) == len(outputs):
                        refined_age_tmp = lb_age.min()

                    refine_list.append(refined_age_tmp)
                # print("refine_list", refine_list)
                # print("reg_num_list", reg_num_list)
                refined_age = np.array(refine_list).mean()

                if (max_local_iter == (iteration + 1)) or (int(refined_age + 0.5) == age):
                    age = int(refined_age + 0.5)
                    memory[i, iteration:] = age
                    pred_age_final.append(age)

                    lb_final_total.append(lb_final)
                    up_final_total.append(up_final)
                    reg_idx.append(reg_idx_final)
                    break
                else:
                    age = int(refined_age + 0.5)
                    memory[i, iteration] = age
                    reg_num_list = []
                    refine_list = []
                    
                    # print("age", age, "reg_num", reg_num)
                    for tmp in range(reg_num):
                        # print("Range: ", np.arange(reg_bound[tmp][0], reg_bound[tmp][1] + 1))
                        if age in np.arange(reg_bound[tmp][0], reg_bound[tmp][1] + 1):
                            reg_num_list.append(tmp)
                            # print("add", tmp)
                    # print("reg_num_list", reg_num_list)
                    iteration += 1

    return pred_age_final

i = 0

def age_estimate_mwr(model, img, face_detect = face_detect, save_folder='static/image', img_size=224, imagenet_stats = imagenet_stats, m = torch.nn.Softmax(dim=1), age_range=torch.arange(21, 61).float()):
    global i
    face_img, image_viz, box = face_detect(img)
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
        
        # change path
        reg_bound = [[21, 28], [24, 33], [29, 40], [34, 49], [41, 60]]
        train_data = pd.read_csv("utk_coral_sampled_5p.csv")
        # test_data = pd.read_csv(os.path.join(arg.data_path, 'utk/UTK_test_coral.csv'))
        test_data = [0]
        
        features_test = [[] for _ in range(5)]
        for reg_idx in range(5):
            outputs = torch.squeeze(model('extraction', x=img, idx=reg_idx))
            outputs_numpy = outputs.cpu().detach().numpy().reshape(-1, 512)
            features_test[reg_idx].extend(outputs_numpy)
            
        features_test = np.array(features_test)
        features = {
                    'train': np.load("train_features.np"), 
                    'test': features_test
                    }
        init_pred = 30 #initial_prediction(train_data, 1, features, 5)
        index_y2_total = get_random_pairs_local_regression(train_data, reg_bound)
        refer_idx, refer_idx_pair = select_random_reference_local_regression(train_data, index_y2_total, reg_bound, limit=1)
        
        age_predict = MWR_local_regression(train_data, test_data, features, refer_idx, refer_idx_pair, [init_pred], reg_bound, model, device)
            
        image_viz = cv2.putText(image_viz,str(age_predict[0]), (box[0]+int(box[2]/2)-30,box[1]+int(box[3])-50),cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 4, cv2.LINE_AA)
        # save_path = os.path.join(save_folder, img_path)
        image_viz = cv2.cvtColor(image_viz, cv2.COLOR_RGB2BGR)
        print(str(age_predict))
        
        save_path = os.path.join(save_folder, str(i) + "render.jpg")
        i += 1
        # image_viz = cv2.cvtColor(image_viz, cv2.COLOR_RGB2BGR)
        print(str(age_predict[0]))
        # image_viz = cv2.resize(image_viz, (470, 470), interpolation = cv2.INTER_AREA)
        cv2.imwrite(save_path, image_viz)
        # cv2.imwrite(save_path, image_viz)
        # plt.imshow(image_viz)
        return save_path
    else:
        print('Cant estimate age, an error occurs!')
        return None
    
model = load_mwr_model(None)

@app.route('/get-age', methods=['POST'])
def age():
    img = request.data
    # img = img.decode("utf-8")
    
    # age_estimate_mwr(model, img_path = './ronaldo.jpg')
    image = Image.open(io.BytesIO(img))

    open_cv_image = np.array(image) 
   # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    path = age_estimate_mwr(model, img=open_cv_image)

    return path
 
if __name__ == '__main__':
   app.run(debug=True)
   

   