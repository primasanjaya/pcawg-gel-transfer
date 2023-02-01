import tensorflow as tf
import pdb
import random as rn
import numpy as np 
import os 

import math 
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping 
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

import skopt 
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer 
from skopt.plots import plot_convergence 
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report,precision_recall_fscore_support,accuracy_score
from scipy import stats
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import sys
	
def to_table(report):
    report = report.splitlines()
    res = []
    res.append(['']+report[0].split())
    for row in report[2:-2]:
        res.append(row.split())
    lr = report[-1].split()
    res.append([' '.join(lr[:3])]+lr[3:])
    return np.array(res)

if __name__ == '__main__': 
	
    fold = int(sys.argv[1])
    feature_type = sys.argv[2]
    wesorwgs = sys.argv[3]

    #pdb.set_trace()

    os.makedirs('./' + feature_type[0:-4], exist_ok=True)
    os.makedirs('./' + feature_type[0:-4]+'/crossvalidation_results', exist_ok=True)
    path_best_model = './{:}/crossvalidation{:.0f}_best_model.keras'.format(feature_type[0:-4],fold)
    best_accuracy = 0 
    data = pd.read_csv(feature_type,index_col = 0)
    data = data.dropna()

    ### Making training, test, validation data 
    if wesorwgs == 'wes':
        num_classes = 13
        training_samples = pd.read_csv('./trainwes.csv',index_col=0)
        validation_samples = pd.read_csv('./valwes.csv',index_col=0)
        test_samples = validation_samples

    elif wesorwgs == 'fullpcawg4':
        num_classes = 24
        training_samples = pd.read_csv('./'+ feature_type[0:-4] +'_train4folds.csv',index_col=0)
        validation_samples = pd.read_csv('./'+ feature_type[0:-4] +'_val4folds.csv',index_col=0)
        test_samples = validation_samples
        
    elif wesorwgs == 'fullpcawg1504':
        num_classes = 24
        training_samples = pd.read_csv('./trainfullpcawg150-4folds.csv',index_col=0)
        validation_samples = pd.read_csv('./valfullpcawg150-4folds.csv',index_col=0)
        test_samples = validation_samples
        
    elif wesorwgs == 'fullpcawg':
        num_classes = 24
        training_samples = pd.read_csv('./trainfullpcawg.csv',index_col=0)
        validation_samples = pd.read_csv('./valfullpcawg.csv',index_col=0)
        test_samples = validation_samples

    elif wesorwgs == 'fulltcga4':
        num_classes = 23
        training_samples = pd.read_csv('./'+ feature_type[0:-4] +'_train4folds.csv',index_col=0)
        validation_samples = pd.read_csv('./'+ feature_type[0:-4] +'_val4folds.csv',index_col=0)
        test_samples = validation_samples
        
    elif wesorwgs == 'fulltcga':
        num_classes = 23
        training_samples = pd.read_csv('./trainfulltcga.csv',index_col=0)
        validation_samples = pd.read_csv('./valfulltcga.csv',index_col=0)
        test_samples = validation_samples

    elif wesorwgs == 'fulltcga10':
        num_classes = 23
        training_samples = pd.read_csv('./'+ feature_type[0:-4] +'_train10folds.csv',index_col=0)
        validation_samples = pd.read_csv('./'+ feature_type[0:-4] +'_val10folds.csv',index_col=0)
        test_samples = validation_samples

    elif wesorwgs == 'fulltcga4_20cls':
        num_classes = 20
        training_samples = pd.read_csv('./'+ feature_type[0:-4] +'_train4folds.csv',index_col=0)
        validation_samples = pd.read_csv('./'+ feature_type[0:-4] +'_val4folds.csv',index_col=0)

        training_samples = training_samples.loc[~training_samples['nm_class'].isin(['CHOL','MESO','READ'])]
        validation_samples = validation_samples.loc[~validation_samples['nm_class'].isin(['CHOL','MESO','READ'])]

        test_samples = validation_samples

    else:
        num_classes = 13
        training_samples = pd.read_csv('./trainwgs.csv',index_col=0)
        validation_samples = pd.read_csv('./valwgs.csv',index_col=0)
        test_samples = validation_samples


    training_samples = training_samples[training_samples.fold == fold]
    training_data = data.loc[data['samples'].isin(training_samples['samples'])]

    validation_samples = validation_samples[validation_samples.fold == fold]
    validation_data = data.loc[data['samples'].isin(validation_samples['samples'])]

    test_samples = test_samples[test_samples.fold == fold]
    test_data = data.loc[data['samples'].isin(test_samples['samples'])]

    metadata_train = training_data.filter(['samples','nm_class'])
    metadata_validation = validation_data.filter(['samples','nm_class'])
    metadata_test = test_data.filter(['samples','nm_class'])

    try:
        training_data = training_data.drop(['samples','nm_class','fold'],axis=1)
        validation_data = validation_data.drop(['samples','nm_class','fold'],axis=1)
        test_data = test_data.drop(['samples','nm_class','fold'],axis=1)
    except:
        training_data = training_data.drop(['samples','nm_class'],axis=1)
        validation_data = validation_data.drop(['samples','nm_class'],axis=1)
        test_data = test_data.drop(['samples','nm_class'],axis=1)

    x_train = training_data.values
    y_train = metadata_train['nm_class'].values

    x_val = validation_data.values
    y_val = metadata_validation['nm_class'].values

    x_test = test_data.values
    y_test = metadata_test['nm_class'].values

    encoder = LabelEncoder()
    test_labels_names = y_test
    y_test = encoder.fit_transform(y_test)
    test_labels = y_test

    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_train = encoder.fit_transform(y_train)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = encoder.fit_transform(y_val)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    validation_data = (x_val, y_val)

    input_size = x_train.shape[1]

    ### Run Bayesian optimization 
    regressor = RandomForestRegressor()
    sc = StandardScaler()
    data_train = sc.fit_transform(x_train)
    data_val = sc.transform(x_test)

    regressor.fit(data_train, y_train)

    Y_pred = regressor.predict(data_val)

    y_pred = np.argmax(Y_pred, axis = 1)

    k = 3 #top-3
    # The following is the calculation method
    #sort
    #max_k_preds3 = Y_pred.argsort(axis=1)

    try:
        max_k_preds3 = Y_pred.argsort(axis=1)[:, -k:][:, ::-1]

        k = 5 #top-3
        # The following is the calculation method
        max_k_preds5 = Y_pred.argsort(axis=1)[:, -k:][:, ::-1]

        real = encoder.fit_transform(test_labels_names)

        correct3 = 0
        correct5 = 0
        for idx,ac in enumerate(real):
            if ac in max_k_preds3[idx]:
                correct3 = correct3 + 1
            if ac in max_k_preds5[idx]:
                correct5 = correct5 + 1

        top3 = correct3 / max_k_preds3.shape[0]
        top5 = correct5 / max_k_preds5.shape[0]
    except:
        pdb.set_trace()

    a = pd.Series(test_labels_names)
    b = pd.Series(test_labels)
    d = pd.DataFrame({'Factor':b, 'Cancer':a})
    d = d.drop_duplicates()
    d = d.sort_values('Factor') 

    p_df = pd.DataFrame(data = Y_pred, columns = d.Cancer, index = test_labels_names)

    ## Generate Confusion Matrix
    c_matrix = confusion_matrix(test_labels, y_pred)
    c_df = pd.DataFrame(data=c_matrix, index=d.Cancer, columns = d.Cancer)

    ## Generate Class Report

    c_report = precision_recall_fscore_support(test_labels, y_pred)
    pd_r = pd.DataFrame(data=c_report).T

    
    pd_r = pd_r.rename(columns={0: "precision", 1: "recall",2: "f1",3: "support"})
    pd_r['accuracy'] = accuracy_score(test_labels, y_pred)
    pd_r['top3']= top3
    pd_r['top5']= top5
    
    if wesorwgs == 'fulltcga4_20cls':
        pd_r.to_csv('./{:}/twenty_RFclass_report_fold{:.0f}.csv'.format(feature_type[0:-4],fold))
        c_df.to_csv('./{:}/twenty_RFconfusion_matrix_fold_{:.0f}.csv'.format(feature_type[0:-4],fold))
        p_df.to_csv('./{:}/twenty_RFprobability_classification_{:.0f}.csv'.format(feature_type[0:-4],fold))
    elif wesorwgs == 'fulltcga':
        os.makedirs('./' + feature_type[0:-4]+'/RF10folds/', exist_ok=True)
        pd_r.to_csv('./{:}/RF10folds/RFclass_report_fold{:.0f}.csv'.format(feature_type[0:-4],fold))
        c_df.to_csv('./{:}/RF10folds/RFconfusion_matrix_fold_{:.0f}.csv'.format(feature_type[0:-4],fold))
        p_df.to_csv('./{:}/RF10folds/RFprobability_classification_{:.0f}.csv'.format(feature_type[0:-4],fold))
    else:
        pd_r.to_csv('./{:}/RFclass_report_fold{:.0f}.csv'.format(feature_type[0:-4],fold))
        c_df.to_csv('./{:}/RFconfusion_matrix_fold_{:.0f}.csv'.format(feature_type[0:-4],fold))
        p_df.to_csv('./{:}/RFprobability_classification_{:.0f}.csv'.format(feature_type[0:-4],fold))
   

    


