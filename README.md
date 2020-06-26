1.Install python 3.5, keras 2.1.3 and Matlab 2019a (1h is needed to intall all softwares)

2.Download the all the files from  https://github.com/lungproject/lungio, and the trained model (lungIO.hdf5) was in the folder named model.

3.Downlad all the data from https://pan.baidu.com/s/1SSN_NrErYckkn-nAyIv9FA  (code: woiu). And excted all the file into the folder named Alldata.

3.Run test_allpatient.py with python to generate the prediction results of all the input slices. (35 seconds are needed to run all the data)

5.Run Obtainresults.m with Matlab to obtain the predicted DLS of each patient. Other test codes were also provided in this file(25 seconds are needed to obtain the final results)
