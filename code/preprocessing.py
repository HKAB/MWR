# Use mtcnn for aligning faces

from mtcnn import MTCNN
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os


train_csv = 'UTK_mtcnn/csv_files/UTK_train_coral.csv'
test_csv = 'UTK_mtcnn/csv_files/UTK_test_coral.csv'

thickness = 2
color = (255, 0, 0)
def face_detect(index, file_path, save_folder):
  path = os.path.join('UTK_new/UTKFace_wild', file_path)
  try:
    img = cv2.imread(path)
    if len(img.shape)==2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face by MTCNN
    results = detector.detect_faces(img)
    if results == []:
      save_path = os.path.join(save_folder, file_path)
      cv2.imwrite(save_path, img)
      print('No detect!')
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

      # image = cv2.rectangle(img, (x,y), (x_end, y_end), color, thickness)
      image = img[y: y_end, x:x_end]
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      save_path = os.path.join(save_folder, file_path)
      cv2.imwrite(save_path, image)
      print('Success:', index, save_path)
  except:
    print('Cant read:', path)

if __name__ == '__main__':
    detector = MTCNN()
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    
    # Processing train set
    save_folder = 'UTK_mtcnn/train'
    for index, row in train_df.iterrows():
        file_path = row['file_path']
        face_detect(index, file_path, save_folder)
    
    # Processing test set
    save_folder = 'UTK_mtcnn/test'
    for index, row in test_df.iterrows():
        file_path = row['file_path']
        face_detect(index, file_path, save_folder)

    print('All have been done!')