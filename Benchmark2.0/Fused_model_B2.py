import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import xgboost
import lightgbm
import joblib
from sklearn.model_selection import train_test_split
from tensorflow import keras
import glob
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


def Load_features():
    with open('Benchmark2.0/Features_proteome.pkl', "rb") as proteome:
        data = pickle.load(proteome)
        Ifeature = np.array(list(data.values())).T
    Label = np.concatenate((np.ones([1482], dtype=int), np.zeros([1482], dtype=int), 
    	                    np.ones([371], dtype=int), np.zeros([371], dtype=int)), axis=0)
    x_train_test, Ind_features, y_train_test, Ind_Label = train_test_split(Ifeature, Label, test_size=0.2, random_state=666)
    x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test, test_size=0.2, random_state=666)
    features_packed = (x_train[:, :1024], 
                       x_train[:, 1024:1024+1536], 
                       x_train[:, 1024+1536:1024+1536+1024],
                       x_train[:, 1024+1536+1024:])
    features_packed_test = (x_test[:, :1024], 
                            x_test[:, 1024:1024+1536], 
                            x_test[:, 1024+1536:1024+1536+1024],
                            x_test[:, 1024+1536+1024:])
    features_packed_ind_test = (Ind_features[:, :1024],
                                Ind_features[:, 1024:1024+1536],
                                Ind_features[:, 1024+1536:1024+1536+1024],
                                Ind_features[:, 1024+1536+1024:])
    return features_packed, y_train, features_packed_test, y_test, features_packed_ind_test, Ind_Label


def Base_estimators(features_packed, Label):
    model1 = lightgbm.LGBMClassifier()
    model2 = xgboost.XGBClassifier()
    model3 = AdaBoostClassifier()
    model4 = RandomForestClassifier()
    model5 = BaggingClassifier()
    for i in range(len(features_packed)):
        Ifeature = features_packed[i]
        j = 0
        for model in (model1, model2, model3, model4, model5):
            model.fit(Ifeature, Label)
            joblib.dump(model, 'Benchmark2.0/First_Model/'+str(i+1)+'_'+str(j+1)+'.pkl')
            j += 1


def data_transfomation(features_packed_test):
    Mydir = sorted(glob.glob('Benchmark2.0/First_Model/*.pkl'))
    x_test_pre = []
    i = 0
    for dir in Mydir:
        print(i, dir)
        model = joblib.load(dir)
        x_test_pro = model.predict_proba(features_packed_test[int(dir[25]) - 1])
        x_test_pre.append(x_test_pro)
        i += 1
    x_test_pre = np.array(x_test_pre)
    x_test_pre = np.transpose(x_test_pre, (2, 0, 1))[1]
    x_test_pre = np.transpose(x_test_pre, (1, 0))
    print(x_test_pre.shape)
    return x_test_pre


def Second_estimators(features_packed_test, y_test):
    x_test_pre = data_transfomation(features_packed_test)
    print(x_test_pre.shape, y_test.shape)
    model1 = lightgbm.LGBMClassifier()
    model2 = xgboost.XGBClassifier()
    model3 = AdaBoostClassifier()
    model4 = RandomForestClassifier()
    model5 = BaggingClassifier()
    i = 0
    for model in (model1, model2, model3, model4, model5):
        model.fit(x_test_pre, y_test)
        joblib.dump(model, 'Benchmark2.0/Second_Model/'+str(i + 1) + '_' + '.h5')
        i += 1


def Independent_test(features_packed_ind_test, Ind_Label):
    x_test_pre = data_transfomation(features_packed_ind_test)
    Test_label = Ind_Label
    Mydir = sorted(glob.glob('Benchmark2.0/Second_Model/*.h5'))
    for dir in Mydir:
        print(dir)
        model = joblib.load(dir)
        Pre_label = model.predict(x_test_pre)
        Acc = metrics.accuracy_score(Test_label, Pre_label)
        MCC = metrics.matthews_corrcoef(Test_label, Pre_label)
        CM = metrics.confusion_matrix(Test_label, Pre_label)
        Pre_label_prob = model.predict_proba(x_test_pre)
        auROC = metrics.roc_auc_score(Test_label, Pre_label_prob[:, 1])
        Spec = CM[0][0] / (CM[0][0] + CM[0][1])
        Sens = CM[1][1] / (CM[1][0] + CM[1][1])
        print('Accuracy:', Acc, " Sensitivity", Sens, " Specificity", Spec, "MCC", MCC, "auROC", auROC)


if __name__ == '__main__':
    features_packed, y_train, features_packed_test, y_test, features_packed_ind_test, Ind_Label = Load_features()
    ###### Train
    Base_estimators(features_packed, y_train)
    Second_estimators(features_packed_test, y_test)
    ###### Test
    Independent_test(features_packed_ind_test, Ind_Label)
