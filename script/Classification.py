# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:25:07 2018

@author: Libo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
#


data = pd.read_csv('../input/cleaned_data.csv',index_col=0)
data['Date'] = pd.to_datetime(data['Date'])

sprays = ['1week_halfmile','1week_1mile', '1week_5mile', '1month_halfmile', '1month_1mile',
       '1month_5mile', '1quarter_halfmile', '1quarter_1mile','1quarter_5mile']
Xsprays = []
for s in sprays:
    spray_var = ['1week_halfmile','1week_1mile', '1week_5mile', '1month_halfmile', '1month_1mile',
       '1month_5mile', '1quarter_halfmile', '1quarter_1mile','1quarter_5mile']  
    X_spray = data.drop(['Address','Block','Street','Trap',
           'AddressNumberAndStreet','AddressAccuracy','weight_1',
          'weight_2','Date','NumMosquitos'],axis=1)
    spray_var.remove(s)
    X_spray = X_spray.drop(spray_var,axis=1)
    Xsprays.append(X_spray)
# add no spray data
spray_var = ['1week_halfmile','1week_1mile', '1week_5mile', '1month_halfmile', '1month_1mile',
   '1month_5mile', '1quarter_halfmile', '1quarter_1mile','1quarter_5mile']
X_spray = data.drop(['Address','Block','Street','Trap',
       'AddressNumberAndStreet','AddressAccuracy','weight_1',
      'weight_2','Date','NumMosquitos'],axis=1)
X_spray = X_spray.drop(spray_var,axis=1)
Xsprays.append(X_spray)


from sklearn.model_selection import train_test_split

X_trains = []
X_tests = []
y_trains = []
y_tests = []

for i in Xsprays:
    X = i.drop('WnvPresent',axis=1)
    Y = i['WnvPresent']

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=20180309,shuffle=True)
    
    X_trains.append(X_train)
    X_tests.append(X_test)
    y_trains.append(y_train)
    y_tests.append(y_test)   

for i in range(len(Xsprays)):
    print(X_trains[i].shape)
    print(X_tests[i].shape)
    print(y_trains[i].shape)
    print(y_tests[i].shape)

Xns = data.drop('WnvPresent',axis=1)
Xns = Xns[['dayofyear',        'Species_CULEX ERRATICUS',
                'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
               'Species_CULEX RESTUANS',       'Species_CULEX SALINARIUS',
               'Species_CULEX TARSALIS',        'Species_CULEX TERRITANS',
                                 'Tmax',                           'Tmin',
                                 'Tavg',                         'Depart',
                             'DewPoint',                        'WetBulb',
                                 'Heat',                           'Cool',
                                'Depth',                       'SnowFall',
                          'PrecipTotal',                    'StnPressure',
                             'SeaLevel',                    'ResultSpeed',
                            'ResultDir',                       'AvgSpeed',]]
Yns = data['WnvPresent']

Xns_train,Xns_test,yns_train,yns_test = train_test_split(Xns,Yns,test_size=0.3,random_state=20180309,shuffle=True)

print('T/T/S report for {}'.format('Data without Spraying'))
print('overall shape: {} \n'.format(data.shape))
print('X shape: {}'.format(Xns.shape))
print('X_train shape: {}'.format(Xns_train.shape))
print('X_test shape: {} \n'.format(Xns_test.shape))
print('Y shape: {}'.format(Yns.shape))
print('Y_train shape: {}'.format(yns_train.shape))
print('Y_test shape: {}'.format(yns_test.shape))

def aucroc(probas,y_true,step=0.01):  #,metric='sensitivity',threshold=95
    obs = y_true.values

    sensitivity = []
    specificity = []

    for t in np.arange(0,1,step): #iterate through each step of classification threshold
        
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        
        for i in range(len(y_true)): #iterate through each observation
            predictions = probas[:,1] > t #only predicted class probability

            ##classify each based on whether correctly predicted
            if predictions[i] == 1 and obs[i] == 1:
                TP += 1
            elif predictions[i] == 0 and obs[i] == 1:
                FN += 1
            elif predictions[i] == 1 and obs[i] == 0:
                FP += 1
            elif predictions[i] == 0 and obs[i] == 0:
                TN += 1
        
        #calculate each metric
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)

        #append all metrics to list 
        sensitivity.append(sens)
        specificity.append(1 - spec)

    #graph sens vs spec curve
    plt.rcParams['font.size'] = 14
    plt.plot(specificity,sensitivity)
    plt.plot([0,1],[0,1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Receiver Operating Characteristic')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    
#随机森林
def randomf(X_train,y_train,X_test,y_test):
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced_subsample',
                                 random_state=20180309,n_jobs=-1,verbose=1)
    print('Cross-validating...')
    rfc_scores = cross_val_score(rfc,X_train,y_train,
                                 cv=3,n_jobs=-1,verbose=1)
    print(rfc_scores)
    print('cross val score: {}'.format(np.mean(rfc_scores)))

    print('Fitting...')
    rfc = rfc.fit(X_train,y_train)

    rfc_preds = rfc.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test,rfc_preds).ravel()
    print('tn,fp,fn,tp')
    print(tn,fp,fn,tp)

    print('accuracy score: {}'.format(accuracy_score(y_test, rfc_preds)))
    print('precision score: {}'.format(precision_score(y_test,rfc_preds)))
    print('recall score: {}'.format(accuracy_score(y_test,rfc_preds)))
    print(classification_report(y_test,rfc_preds))

    rfc_probas = rfc.predict_proba(X_test)
    print(roc_auc_score(y_test,rfc_probas[:,1]))

    aucroc(rfc_probas,y_test)

def svm(X_train,y_train,X_test,y_test):
    print('GridSearching...')
    svm_params = {'kernel':['linear','rbf','sigmoid'],
                  }

    svm = SVC(kernel='kernel',class_weight='balanced',
              verbose=1,random_state=20180309,probability=True,max_iter=10000)
    svm_gscv = GridSearchCV(svm,svm_params,
                            n_jobs=-1,verbose=1,cv=3)
    print('Fitting...')
    svm_gscv.fit(X_train,y_train)

    print('Scoring...')
    svm_scores = svm_gscv.score(X_train,y_train)

    print(svm_scores)
    print('accuracy score: {}'.format(np.mean(svm_scores)))

    svm_preds = svm_gscv.best_estimator_.predict(X_test)
    svm_probas = svm_gscv.best_estimator_.predict_proba(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test,svm_preds).ravel()
    print('tn,fp,fn,tp')
    print(tn,fp,fn,tp)

    print('accuracy score: {}:'.format(accuracy_score(y_test, svm_preds)))
    print('precision score: {}'.format(precision_score(y_test,svm_preds)))
    print('recall score: {}'.format(accuracy_score(y_test,svm_preds)))
    print(classification_report(y_test,svm_preds))

    print(roc_auc_score(y_test,svm_probas[:,1]))

    aucroc(svm_probas,y_test)
    
    
if __name__ == "__main__":    
    randomf(X_train,y_train,X_test,y_test)