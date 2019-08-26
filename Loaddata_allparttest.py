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



def load_traindata():
    
    img = np.load("./Alldata/xptrainpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/xptrainct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/xptrainfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/yptrain.npy")
    label = np.asarray(img,dtype="float32")
    return datapet,datact,datafuse,label
    

def load_testdata():
    
    img = np.load("./Alldata/xptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/xptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/xptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/yptest.npy")
    label = np.asarray(img,dtype="float32")
    return datapet,datact,datafuse,label


def load_hlmpdltestdata():
    
    img = np.load("./Alldata/hlmpdlxptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmpdlxptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmpdlxptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/hlmpdlyp.npy")
    label = np.asarray(img,dtype="float32")
    return datapet,datact,datafuse,label



def load_hlmtestdata():
    
    img = np.load("./Alldata/hlmxptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmxptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmxptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")

    return datapet,datact,datafuse


def load_hlmvaldata():
    
    img = np.load("./Alldata/hlmvalxptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmvalxptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmvalxptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")

    return datapet,datact,datafuse    
