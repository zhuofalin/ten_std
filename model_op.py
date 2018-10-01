#! /usr/bin/python
# _*_ coding:utf-8 _*_

from sklearn.externals import joblib
from sklearn import datasets, metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import scorer
import scipy.io as sio
import time
import numpy as np
import matplotlib.pyplot as plt


Par_dir = r"E:\zflPro\ParandAcc"
# model_name = 'SVMtrain_model.m'
def Get_time_dif (start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def result_metrics(bayesclf,bayespredicted,TrainSample,Trainlabel,Testlabel):
    print("Classification report for classifier %s:\n%s\n" % (bayesclf, metrics.classification_report(Testlabel, bayespredicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(Testlabel, bayespredicted))
    print("score:", bayesclf.score(TrainSample, Trainlabel))
    print("accuracy_score:", accuracy_score(Testlabel, bayespredicted))
    return 0

def Save_Model(bayesclf, model_dir, model_ID):
    save_model = str(model_dir)+"\\"+str(model_ID)+"train_model.m"
    joblib.dump(bayesclf, save_model)
    model_end = str(model_ID)+" finished"
    print(model_end)
    return 0

def Save_Weight(model_ID, name, Valuerange, Acclist, each_classAcc):
    result = {}
    result['Valuerange'] = Valuerange
    result['Acclist'] = Acclist
    result['each_classAcc'] = each_classAcc
    filedir = Par_dir +"\\" + str(model_ID) + str(name) + "parandacc.txt"
    parfile = open (filedir,'w')
    parfile.write("parameters_name:" + str(name) + '\n')
    parfile.write(str(name) + ' ' + str("Acclist") + ' ' + str("each_classAcc") + '\n')

    # plt.figure(figsize=(8, 4))
    # plt.plot(Valuerange, Acclist, linewidth=2)
    # plt.xlabel("Valuerange")
    # plt.ylabel("Acclist")
    # plt.ylim(-1.2, +1.2)
    # plt.show()

    for i in range(len(Valuerange)):
        parfile.write(str(result['Valuerange'][i])+  '  ' + str(result['Acclist'][i]) +  '  ' + str(result['each_classAcc'][i]))
        parfile.write('\n')
    parfile.close()
    return 0




