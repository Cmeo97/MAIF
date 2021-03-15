#!/usr/bin/env python3.7
import torch
import numpy as np
import cv2

class Dataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,list_IDs, labels):

        self.labels = labels
        self.list_IDs = list_IDs
        self.Im = np.zeros((1, 128, 128))
        self.X_matrix = np.zeros((500100, 7))
        # Opening files
        x_file = open('JointStates_q.txt', 'r+')

        # Reading from the file
        content_x = x_file.readlines()

        # Definition of the matrices
        x_matrix = []
        dim = 500000
        for line in content_x:
            x_matrix = np.matrix(line, float)
            dim_x = x_matrix.shape


        c=0
        #self.X_matrix = np.zeros((850000,7))
        for i in range(dim):
           for l in range(7):
               self.X_matrix[i,l] = x_matrix[0,c]
               c=c+1
        print(self.X_matrix.shape)

        x_file.close()


    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, idx):

       i = 0
       j = idx
       img_path = 'poses/camera_image_' + str(j) + '.jpeg'
       image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
       while (image is None):
          img_path = 'poses/camera_image_' + str(j + i) + '.jpeg'
          image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
          i += -1
          print(i)

       self.Im[0,:,:] = image.astype("float32")/255

       joints = self.X_matrix[idx,:]

       joints[0] += 2.5
       joints[1] += 2.5
       joints[2] += 1.3
       joints[3] += 4.2
       joints[4] += 1.3
       joints[5] += 1.2
       joints[6] += 2.5

       return joints, self.Im




