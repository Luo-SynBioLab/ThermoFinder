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
    fea_1 = np.array(pd.read_excel('Ensemle_models/Features_protTrans.xlsx'))   # 1024
    fea_2 = np.array(pd.read_excel('Ensemle_models/Features_CPC.xlsx'))   # 1536
    fea_3 = np.array(pd.read_excel('Ensemle_models/Features_Elmo.xlsx'))   # 1024
    fea_4 = np.array(pd.read_excel('Ensemle_models/Features_CNN.xlsx'))   # 1100
    features = np.concatenate((fea_1, fea_2, fea_3, fea_4), axis=1)[:2242]
    Ind_features = np.concatenate((fea_1, fea_2, fea_3, fea_4), axis=1)[2242:]
    features_packed = (features[:, :1024],
                       features[:, 1024:1024+1536],
                       features[:, 1024+1536:1024+1536+1024],
                       features[:, 1024+1536+1024:1024+1536+1024+1100])
    features_packed_ind_test = (Ind_features[:, :1024],
                                Ind_features[:, 1024:1024+1536],
                                Ind_features[:, 1024+1536:1024+1536+1024],
                                Ind_features[:, 1024+1536+1024:1024+1536+1024+1100])
    Label = np.concatenate((np.zeros([1151], dtype=int), np.ones([1091], dtype=int)), axis=0)
    Ind_Label = np.concatenate((np.zeros([289], dtype=int), np.ones([273], dtype=int)), axis=0)
    return features_packed, Label, features_packed_ind_test, Ind_Label


def Independent_test(features_packed, Label, features_packed_ind_test, Ind_Label):
    model1 = lightgbm.LGBMClassifier()
    model2 = xgboost.XGBClassifier()
    model3 = AdaBoostClassifier()
    model4 = RandomForestClassifier()
    model5 = BaggingClassifier()
    Peptide_data = np.zeros([4, 5, 5], dtype=float)
    for i in range(4):
        j = 0
        for model in (model1, model2, model3, model4, model5):
            features_packed_sub = features_packed[i]
            x_test_pre = features_packed_ind_test[i]
            Test_label = Ind_Label
            model.fit(features_packed_sub, Label)
            Pre_label = model.predict(x_test_pre)
            Acc = metrics.accuracy_score(Test_label, Pre_label)
            MCC = metrics.matthews_corrcoef(Test_label, Pre_label)
            CM = metrics.confusion_matrix(Test_label, Pre_label)
            Pre_label_prob = model.predict_proba(x_test_pre)
            auROC = metrics.roc_auc_score(Test_label, Pre_label_prob[:, 1])
            Spec = CM[0][0] / (CM[0][0] + CM[0][1])
            Sens = CM[1][1] / (CM[1][0] + CM[1][1])
            print('Accuracy:', Acc, " Sensitivity", Sens, " Specificity", Spec, "MCC", MCC, "auROC", auROC)
            Peptide_data[i][j][0] = Acc
            Peptide_data[i][j][1] = Sens
            Peptide_data[i][j][2] = Spec
            Peptide_data[i][j][3] = MCC
            Peptide_data[i][j][4] = auROC
            j += 1
    data = Peptide_data.reshape((20, 5)).T
    res = pd.DataFrame({"Accuracy:": data[0], " Sensitivity": data[1], " Specificity": data[2],
                        "MCC": data[3], "auROC": data[4]})
    res.to_excel('Benchmark1.0/Single_model_feature_B1.xlsx')


if __name__ == '__main__':
    features_packed, Label, features_packed_ind_test, Ind_Label = Load_features()
    Independent_test(features_packed, Label, features_packed_ind_test, Ind_Label)
