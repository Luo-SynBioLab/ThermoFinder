import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost
import lightgbm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
import pickle
import glob
import os


def Load_features():
    with open('Features_All_accurate.pkl', "rb") as proteome:
        data = pickle.load(proteome)
        Ifeature = np.array(list(data.values())).T
    Label = []
    with open('Enzyme_Sequence.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] == '>':
                Label.append(float(line[line.index('_')+1:-1]))
    Label = np.array(Label)
    x_train_test, Ind_features, y_train_test, Ind_Label = train_test_split(Ifeature, Label, test_size=0.2, random_state=666)
    x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test, test_size=0.2, random_state=666)

    features_packed = (x_train[:, :1100], 
    	               x_train[:, 1100:1100 + 1024], 
    	               x_train[:, 1100 + 1024:1100 + 1024+1536],
                       x_train[:, 1100 + 1024+1536:])
    features_packed_test = (x_test[:, :1100], 
    	                    x_test[:, 1100:1100 + 1024], 
    	                    x_test[:, 1100 + 1024:1100 + 1024+1536],
                            x_test[:, 1100 + 1024+1536:])
    features_packed_ind_test = (Ind_features[:, :1100],
                                Ind_features[:, 1100:1100 + 1024],
                                Ind_features[:, 1100 + 1024:1100 + 1024+1536],
                                Ind_features[:, 1100 + 1024+1536:])

    return features_packed, y_train, features_packed_test, y_test, features_packed_ind_test, Ind_Label


def Base_estimators(features_packed, Label):
    model1 = lightgbm.LGBMRegressor()
    model2 = xgboost.XGBRegressor()
    model3 = AdaBoostRegressor()
    model4 = RandomForestRegressor()
    model5 = BaggingRegressor()
    for i in range(len(features_packed)):
        Ifeature = features_packed[i]
        j = 0
        for model in (model1, model2, model3, model4, model5):
            model.fit(Ifeature, Label)
            joblib.dump(model, 'ThermoSeq_c1.0/First_Model/'+str(i+1)+'_'+str(j+1)+'.pkl')
            j += 1


def data_transfomation(features_packed_test):
    Mydir = sorted(glob.glob('ThermoSeq_c1.0/First_Model/*.pkl'))
    x_test_pre = []
    i = 0
    for dir in Mydir:
        print(i, dir)
        model = joblib.load(dir)
        x_test_pre.append(model.predict(features_packed_test[int(dir[27]) - 1]))
        i += 1
    x_test_pre = np.array(x_test_pre)
    print(x_test_pre.shape)
    x_test_pre = x_test_pre.T
    print(x_test_pre.shape)
    return x_test_pre


def Second_estimators(features_packed_test, y_test):
    x_test_pre = data_transfomation(features_packed_test)
    print(x_test_pre.shape, y_test.shape)
    model1 = lightgbm.LGBMRegressor()
    model2 = xgboost.XGBRegressor()
    model3 = AdaBoostRegressor()
    model4 = RandomForestRegressor()
    model5 = BaggingRegressor()
    i = 0
    for model in (model1, model2, model3, model4, model5):
        model.fit(x_test_pre, y_test)
        joblib.dump(model, 'ThermoSeq_c1.0/Second_Model/'+str(i + 1) + '_' + '.h5')
        i += 1


def Independent_test(features_packed_ind_test, Ind_Label):
    x_test_pre = data_transfomation(features_packed_ind_test)
    Test_label = Ind_Label
    Mydir = sorted(glob.glob('ThermoSeq_c1.0/Second_Model/*.h5'))
    for dir in Mydir:
        print(dir)
        model = joblib.load(dir)
        Pre_label = model.predict(x_test_pre)
        MSE = np.sqrt(mean_squared_error(Test_label, Pre_label))
        Corr = np.corrcoef(Test_label, Pre_label)[1][0]
        MAE = mean_absolute_error(Test_label, Pre_label)
        print('MSE:', MSE, 'MAE:', MAE, 'Corr:', Corr)


if __name__ == '__main__':
    features_packed, y_train, features_packed_test, y_test, features_packed_ind_test, Ind_Label = Load_features()
    ###### Train
    Base_estimators(features_packed, y_train)
    Second_estimators(features_packed_test, y_test)
    ###### Test
    Independent_test(features_packed_ind_test, Ind_Label)
