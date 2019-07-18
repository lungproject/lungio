import os  
from PIL import Image  
import csv
import numpy as np  
from keras import backend as K
import scipy.io
 


def load_sampledata():

    img = np.load("./data/xpsamplepet.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./data/xpsamplect.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./data/xpsamplefuse.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./data/label.npy")
    label = np.asarray(img,dtype="float32")
    return datapet,datact,datafuse,label

