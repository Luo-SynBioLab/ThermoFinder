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
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


def Load_features():
    fea_1 = np.array(pd.read_excel('Ensemle_models/Features_protTrans.xlsx'))   #1024
    fea_2 = np.array(pd.read_excel('Ensemle_models/Features_CPC.xlsx'))   # 1536
    fea_3 = np.array(pd.read_excel('Ensemle_models/Features_Elmo.xlsx'))   # 1024
    fea_4 = np.array(pd.read_excel('Ensemle_models/Features_CNN.xlsx'))   # 1100
    features = np.concatenate((fea_1, fea_2, fea_3, fea_4), axis=1)[:2242]
    Label = np.concatenate((np.zeros([1151], dtype=int), np.ones([1091], dtype=int)), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(features, Label, test_size=0.2, random_state=666)
    features_packed = (x_train[:, :1024], x_train[:, 1024:1024+1536], x_train[:, 1024+1536:1024+1536+1024],
                       x_train[:, 1024+1536+1024:1024+1536+1024+1100])
    features_packed_test = (x_test[:, :1024], x_test[:, 1024:1024+1536], x_test[:, 1024+1536:1024+1536+1024],
                            x_test[:, 1024+1536+1024:1024+1536+1024+1100])

    Ind_features = np.concatenate((fea_1, fea_2, fea_3, fea_4), axis=1)[2242:]
    features_packed_ind_test = (Ind_features[:, :1024],
                                Ind_features[:, 1024:1024+1536],
                                Ind_features[:, 1024+1536:1024+1536+1024],
                                Ind_features[:, 1024+1536+1024:1024+1536+1024+1100])

    Ind_Label = np.concatenate((np.zeros([289], dtype=int), np.ones([273], dtype=int)), axis=0)
    return features_packed, y_train, features_packed_test, y_test, features_packed_ind_test, Ind_Label


def data_transfomation(features_packed_test, rep):
    Mydir = sorted(glob.glob('Benchmark1.0/First_Model/*_'+str(rep)+'.pkl'))
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
    for rep in range(1, 6):
        x_test_pre = data_transfomation(features_packed_test, rep)
        print(x_test_pre.shape, y_test.shape)
        model1 = lightgbm.LGBMClassifier()
        model2 = xgboost.XGBClassifier()
        model3 = AdaBoostClassifier()
        model4 = RandomForestClassifier()
        model5 = BaggingClassifier()
        i = 0
        for model in (model1, model2, model3, model4, model5):
            model.fit(x_test_pre, y_test)
            joblib.dump(model, 'Benchmark1.0/Models_second_model/'+str(i + 1) + '_' + str(rep) + '.h5')
            i += 1


def Independent_test(features_packed_ind_test, Ind_Label):
    Peptide_data = np.zeros([5, 5, 5], dtype=float)
    for rep in range(1, 6):
        x_test_pre = data_transfomation(features_packed_ind_test, rep)
        Test_label = Ind_Label
        Mydir = sorted(glob.glob('Benchmark1.0/Models_second_model/*_'+str(rep)+'.h5'))
        j = 0
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
            Peptide_data[rep-1][j][0] = Acc
            Peptide_data[rep-1][j][1] = Sens
            Peptide_data[rep-1][j][2] = Spec
            Peptide_data[rep-1][j][3] = MCC
            Peptide_data[rep-1][j][4] = auROC
            j += 1
    data = Peptide_data.reshape((25, 5)).T
    res = pd.DataFrame({"Accuracy:": data[0], " Sensitivity": data[1], " Specificity": data[2],
                        "MCC": data[3], "auROC": data[4]})
    res.to_excel('Benchmark1.0/Single_models_B1.xlsx')


if __name__ == '__main__':
    features_packed, y_train, features_packed_test, y_test, features_packed_ind_test, Ind_Label = Load_features()
    ###### Train
    Second_estimators(features_packed_test, y_test)
    ###### Test
    Independent_test(features_packed_ind_test, Ind_Label)
