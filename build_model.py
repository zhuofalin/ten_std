#! /usr/bin/python
# _*_ coding:utf-8 _*_

from data.data_loader import data_loader
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from model_op import Save_Model, result_metrics, Save_Weight
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, metrics


filename = r'E:\zflPro\data\getfeature-resnet-1-dsn-flatten0.mat'
# filename = r'getfeature-resnet-1-dsn-flatten0.mat'
TrainSample, Trainlabel, TestSample, Testlabel = data_loader(filename)

model_dir  = r'E:\zflPro\MLmodel'
model_TDlist   = ["SVM", "RF", "ADB"]
C_range    = range(1, 20, 1)
gama_max   = 0.001
max_range  = range(500,1000,10)
nesti_ran  = range(500)
depth_range= range(30)
rate_range = 10
njob_range = range(20)

# svm.SVC

def svmtrain(TrainSample, Trainlabel, Testlabel):
    model_TD = str(model_TDlist[0])
    Valuerange = []
    Acclist = []
    each_classAcc = []
    MValuerange = []
    MAcclist = []
    Meach_classAcc = []
    namelist = ["C", "gamma", "maxiter"]

    for cr in C_range:
        Valuerange.append(cr)
        classifier = svm.SVC(C = cr, gamma=0.00001, verbose=False, kernel='rbf', max_iter= 930)
        classifier.fit(TrainSample, Trainlabel)
        predicted = classifier.predict(TestSample)
        accurage  = metrics.accuracy_score(Testlabel, predicted)
        Acclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, predicted, average=None))
        acc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        each_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[0] , Valuerange, Acclist, each_classAcc)
        BestC = C_range[Acclist.index(max(Acclist))]

    for it in max_range:
        MValuerange.append(it)
        classifier = svm.SVC(C= BestC, gamma= 0.00001, verbose= False, kernel='rbf', max_iter= it)
        classifier.fit(TrainSample, Trainlabel)
        predicted = classifier.predict(TestSample)
        accurage = metrics.accuracy_score(Testlabel, predicted)
        MAcclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, predicted, average=None))
        Macc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        Meach_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[2], MValuerange, MAcclist, Meach_classAcc)
        BestM = MValuerange[MAcclist.index(max(MAcclist))]

    classifier =svm.SVC(C= BestC ,gamma= 0.00001,verbose= False,kernel= 'rbf',max_iter= BestM)
    classifier.fit(TrainSample, Trainlabel)
    predicted = classifier.predict(TestSample)
    result_metrics(classifier,  predicted, TrainSample, Trainlabel, Testlabel)
    Save_Model(classifier, model_dir, model_TD)

    return  0


def RFtrain(TrainSample, Trainlabel, Testlabel):
    model_TD = str(model_TDlist[1])
    Valuerange = []
    Acclist = []
    each_classAcc = []
    MValuerange = []
    MAcclist = []
    Meach_classAcc = []
    namelist = ["n_estimators", "n_jobs"]
    for it in nesti_ran:
        Valuerange.append(it)
        RFCclf = RandomForestClassifier(n_estimators= it, criterion="gini", n_jobs= 5)
        RFCclf = RFCclf.fit(TrainSample, Trainlabel)
        RFCpredicted = RFCclf.predict(TestSample)
        accurage = metrics.accuracy_score(Testlabel, RFCpredicted)
        Acclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, RFCpredicted, average=None))
        acc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        each_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[0], Valuerange, Acclist, each_classAcc)
        Bestit = Valuerange[Acclist.index(max(Acclist))]

    for it in njob_range:
        MValuerange.append(it)
        RFCclf = RandomForestClassifier(n_estimators= it, criterion= "gini", n_jobs= 5)
        RFCclf = RFCclf.fit(TrainSample, Trainlabel)
        RFCpredicted = RFCclf.predict(TestSample)
        accurage = metrics.accuracy_score(Testlabel, RFCpredicted)
        MAcclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, RFCpredicted, average= None))
        acc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        Meach_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[1], MValuerange, MAcclist, Meach_classAcc)
        BestM = MValuerange[MAcclist.index(max(MAcclist))]

    RFCclf = RandomForestClassifier(n_estimators= Bestit, criterion= "gini", n_jobs= BestM)
    RFCclf = RFCclf.fit(TrainSample, Trainlabel)
    RFCpredicted = RFCclf.predict(TestSample)
    result_metrics(RFCclf, RFCpredicted, TrainSample, Trainlabel, Testlabel)
    Save_Model(RFCclf, model_dir, model_TD)
    return 0

def ADBtrain(TrainSample, Trainlabel, Testlabel):
    # ADB
    model_TD = str(model_TDlist[2])
    Valuerange = []
    Acclist = []
    each_classAcc = []
    MValuerange = []
    MAcclist = []
    Meach_classAcc = []
    namelist = ["max_depth", "n_estimators", "learning_rat"]
    LValuerange = []
    LAcclist = []
    Leach_classAcc = []
    LR = 0.2

    for dep in depth_range:
        Valuerange.append(dep)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=dep), n_estimators=320, learning_rate=1.2, algorithm= 'SAMME')
        clf.fit(TrainSample, Trainlabel)
        adapredicted = clf.predict(TestSample)
        accurage = metrics.accuracy_score(Testlabel, adapredicted)
        Acclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, adapredicted, average=None))
        acc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        each_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[0], Valuerange, Acclist, each_classAcc)
        Bestdep = depth_range[Acclist.index(max(Acclist))]

    for it in max_range:
        MValuerange.append(it)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=Bestdep), n_estimators=it, learning_rate=1.2, algorithm='SAMME')
        clf.fit(TrainSample, Trainlabel)
        adapredicted = clf.predict(TestSample)
        accurage = metrics.accuracy_score(Testlabel, adapredicted)
        MAcclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, adapredicted, average=None))
        acc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        Meach_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[1], MValuerange, MAcclist, Meach_classAcc)
        BestM = MValuerange[MAcclist.index(max(MAcclist))]

    while(LR < rate_range):
        LValuerange.append(lR)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=Bestdep), n_estimators=BestM, learning_rate=LR, algorithm='SAMME')
        clf.fit(TrainSample, Trainlabel)
        adapredicted = clf.predict(TestSample)
        accurage = metrics.accuracy_score(Testlabel, adapredicted)
        LAcclist.append(accurage)
        acc_for_each = list(metrics.precision_score(Testlabel, adapredicted, average=None))
        acc_for_eachclass = {str(i + 1): acc_for_each for i in range(len(acc_for_each))}
        Leach_classAcc.append(acc_for_eachclass)
        Save_Weight(model_TD, namelist[2], LValuerange, LAcclist, Leach_classAcc)
        BestL = MValuerange[MAcclist.index(max(MAcclist))]

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=Bestdep), n_estimators=BestM, learning_rate=BestL, algorithm='SAMME')
    # clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=300,learning_rate=0.1,algorithm='SAMME.R')
    clf.fit(TrainSample, Trainlabel)
    adapredicted = clf.predict(TestSample)
    result_metrics(clf, adapredicted, TrainSample, Trainlabel, Testlabel)
    Save_Model(clf, model_dir, model_TD)
    return 0

if __name__ == '__main__':
    svmtrain(TrainSample, Trainlabel, Testlabel)
    RFtrain(TrainSample, Trainlabel, Testlabel)
    ADBtrain(TrainSample, Trainlabel, Testlabel)