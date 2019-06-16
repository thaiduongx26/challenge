import os
import numpy as np
import cv2
from scipy.io import loadmat
from tqdm import tqdm

traindir = "data/cars_train/"
testdir = "data/cars_test/"
labelTrainLink = "data/dev_kit/cars_train_annos.mat"
labelTestLink = "data/cars_test_annos_withlabels.mat"
data_processed_dir = "data-processed/"
img_width, img_height = 224, 224

def preprocess_image(image):
    h, w, _ = image.shape
    img_final = image
    if(h > w):
        pad = np.zeros((h, h - w, 3), dtype="int8")
        img_final = np.concatenate((image, pad), axis=1)
    elif(h < w):
        pad = np.zeros((w - h, w, 3), dtype="int8")
        img_final = np.concatenate((image, pad), axis=0)
    img_final = cv2.resize(img_final, (img_width, img_height))
    return img_final

def prepair_data(folderpath, anno, save_res_path, margin=16, type='train'):
    for i in tqdm(range(len(anno["annotations"][0]))):
        img = cv2.imread(folderpath+anno["annotations"][0][i][5][0])
        height, width, _ = img.shape
        margin = margin
        x1 = anno["annotations"][0][i][0][0][0]
        y1 = anno["annotations"][0][i][1][0][0]
        x2 = anno["annotations"][0][i][2][0][0]
        y2 = anno["annotations"][0][i][3][0][0]
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        label = anno["annotations"][0][i][4][0][0]
        filename = anno["annotations"][0][i][5][0]
        img_car = img[y1:y2, x1:x2]
        img = preprocess_image(img)
        img_car = preprocess_image(img_car)
        if not os.path.isdir(save_res_path + str(label)):
            os.mkdir(save_res_path + str(label))
        if(type == 'train'):
            cv2.imwrite(save_res_path + str(label) + "/" + filename.split(".")[0] + "_1" + ".jpg", img)
            cv2.imwrite(save_res_path + str(label) + "/" + filename.split(".")[0] + "_2" + ".jpg", img_car)
        else:
            cv2.imwrite(save_res_path + str(label) + "/" + filename.split(".")[0] + "_3" + ".jpg", img)
            cv2.imwrite(save_res_path + str(label) + "/" + filename.split(".")[0] + "_4" + ".jpg", img_car)

if __name__ == "__main__":
    annotrain = loadmat(labelTrainLink)
    annotest = loadmat(labelTestLink)
    prepair_data(traindir, annotrain, data_processed_dir, margin=5, type='train')
    prepair_data(testdir, annotest, data_processed_dir, margin=5, type='test')