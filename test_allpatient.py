from __future__ import print_function

import datetime
import keras
import numpy as np

import os
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.models import  Sequential
from keras.layers import Input,Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold

from keras import backend as K

from Loaddata_allparttest import load_sampledata,load_traindata,load_testdata,load_hlmpdltestdata,load_hlmtestdata,load_hlmvaldata
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


modelpetct1 = load_model('./model/LungIO.hdf5')#,weightspatient2-improvement-40-0.67
modelpetct1.summary()

xpet_train,xct_train,xfuse_train,y_train = load_traindata()

xpet_train= np.expand_dims(xpet_train, axis=3)
xct_train = np.expand_dims(xct_train, axis=3)
xfuse_train = np.expand_dims(xfuse_train, axis=3)
predicttrainpetct1 = modelpetct1.predict([xpet_train,xct_train,xfuse_train  ], verbose=1)
np.savetxt("./Results/predicttrain.txt",predicttrainpetct1 )


xpet_test,xct_test,xfuse_test,y_test = load_testdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
predicttestpetct1 = modelpetct1.predict([xpet_test,xct_test,xfuse_test  ], verbose=1)
np.savetxt("./Results/predicttest.txt",predicttestpetct1 )

xpet_hlmpdl,xct_hlmpdl,xfuse_hlmpdl,y_hlmpdl = load_hlmpdltestdata()

xpet_hlmpdl= np.expand_dims(xpet_hlmpdl, axis=3)
xct_hlmpdl = np.expand_dims(xct_hlmpdl, axis=3)
xfuse_hlmpdl = np.expand_dims(xfuse_hlmpdl, axis=3)
predicthlmpdlpetct1 = modelpetct1.predict([xpet_hlmpdl,xct_hlmpdl,xfuse_hlmpdl  ], verbose=1)
np.savetxt("./Results/predicthlmpdl.txt",predicthlmpdlpetct1 )



xpet_hlmIO,xct_hlmIO,xfuse_hlmIO = load_hlmtestdata()

xpet_hlmIO= np.expand_dims(xpet_hlmIO, axis=3)
xct_hlmIO = np.expand_dims(xct_hlmIO, axis=3)
xfuse_hlmIO = np.expand_dims(xfuse_hlmIO, axis=3)
predicthlmIOpetct1 = modelpetct1.predict([xpet_hlmIO,xct_hlmIO,xfuse_hlmIO  ], verbose=1)
np.savetxt("./Results/predicthlmIO.txt",predicthlmIOpetct1 )


xpet_hlmval,xct_hlmval,xfuse_hlmval = load_hlmvaldata()

xpet_hlmval= np.expand_dims(xpet_hlmval, axis=3)
xct_hlmval = np.expand_dims(xct_hlmval, axis=3)
xfuse_hlmval = np.expand_dims(xfuse_hlmval, axis=3)
predicthlmvalpetct1 = modelpetct1.predict([xpet_hlmval,xct_hlmval,xfuse_hlmval  ], verbose=1)
np.savetxt("./Results/predicthlmval.txt",predicthlmvalpetct1 )

