#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/lilianapamasa/Desktop/Machine Learning Final Project/Myocardial infarction complications Database.csv', header=0)

df


### PREPROCESSING

# drop columns with more than 20% of missing data
newdf = df.drop(['IBS_NASL', 'S_AD_KBRIG', 'D_AD_KBRIG', 'GIPO_K',
                       'K_BLOOD', 'GIPER_NA', 'NA_BLOOD', 'KFK_BLOOD', 'NA_KB',
                       'NOT_NA_KB', 'LID_KB'], axis=1)
#drop columns not pertaining to time of admission
newdf = newdf.drop(['R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_R_1_n', 'NA_R_2_n', 
                 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n'], axis = 1)

#drop rows with more than 20% of missing values
newdf= newdf.dropna(thresh=99)

newdf

df_categorial = newdf.drop(['AGE', 'S_AD_ORIT','D_AD_ORIT', 
                            'ALT_BLOOD','AST_BLOOD','L_BLOOD',
                            'ROE'], axis = 1)

df_real = newdf.filter(['ID','AGE', 'S_AD_ORIT','D_AD_ORIT', 
                            'ALT_BLOOD','AST_BLOOD','L_BLOOD',
                            'ROE'], axis = 1)


df_real = df_real.fillna(df_real.median())
df_categorial = df_categorial.apply(lambda x: x.fillna(x.value_counts().index[0]))

MIdf = pd.merge(right=df_categorial ,left= df_real, on='ID')

MIdf

MIdf = MIdf.drop(['ID', 'TRENT_S_n','FIBR_PREDS',
                  'PREDS_TAH','JELUD_TAH','FIBR_JELUD',
                  'A_V_BLOK','OTEK_LANC', 'RAZRIV','DRESSLER',
                 'REC_IM', 'P_IM_STEN','LET_IS'], axis = 1)

MIdf
print(MIdf.shape)

X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values


### Decision Tree All Features

#best parameters for DT for all features
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6],
                     'criterion': ['gini', 'entropy'], 
                      'min_samples_leaf': [1,2,3,4,5,6],
                     'random_state' : [1,35,42],
                    'max_features' : ['sqrt', 'log2', None, 'auto']}

GridDT = GridSearchCV(estimator=tree,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
GridDT.fit(X_train,y_train)

print("Best parameters found: ", GridDT.best_params_) 
print("Best accuracy score found: %.3f "% GridDT.best_score_)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(GridDT, X_train, y_train,
                         scoring='accuracy', cv=10)

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()
X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


tree = DecisionTreeClassifier(criterion = 'entropy', max_depth= 2, 
                              max_features= None, min_samples_leaf= 1, 
                              random_state= 1)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

p = sns.heatmap(confmat, annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Decision Tree All Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('DT_all_CM.pdf')

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test, y_pred, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test, y_pred, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))


# ## Random Forest All Features

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

forest = RandomForestClassifier(n_estimators = 500, n_jobs = 2)
forest.fit(X_train, y_train)

param_grid = {'max_depth': [1, 2, 3, 4, 5, 6],
                     'criterion': ['gini', 'entropy'], 
                      'min_samples_leaf': [1,2,3,4,5,6],
                     'random_state' : [1,35,42],
                    'max_features' : ['sqrt', 'log2', None]}

gs = GridSearchCV(estimator=forest,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
gs.fit(X_train, y_train)
print("Best parameters found: ", gs.best_params_) 
print("Best accuracy score found: %.3f "% gs.best_score_)

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

rf = RandomForestClassifier(criterion = 'gini', max_depth = 5,
                           random_state = 1, max_features = None,
                           min_samples_leaf=3, n_estimators=500, n_jobs=2)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

p = sns.heatmap(confmat, annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Random Forest All Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('RT_all_CM.pdf')

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test, y_pred, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test, y_pred, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))


### Feature Selection

import pandas as pd
features = MIdf.columns
f_i = list(zip(features,rf.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.rcParams['figure.figsize'] = [50,50]
plt.show()
plt.savefig('RT_featrank.pdf')

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = features

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                        feat_labels[indices[f]],
                            importances[indices[f]]))


### ROC Decision Tree All Features

import sklearn.metrics as metrics

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth= 2, 
                              max_features= None, min_samples_leaf= 1, 
                              random_state= 1)
tree.fit(X_train,y_train)

y_pred_proba = tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.title('ROC of Decision Tree: All Features', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [5,5]
plt.show()
plt.savefig("DT_all.pdf") 


### ROC Random Forest All Features
rf = RandomForestClassifier(criterion = 'gini', max_depth = 5,
                           random_state = 1, max_features = None,
                           min_samples_leaf=3, n_estimators=500, n_jobs=2)

rf.fit(X_train, y_train)

y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.title('ROC of Random Forest: All Features', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()
plt.savefig("RF_all_ROC.pdf") 


# ## Creating the feature selection dataset
AllFeatdf = MIdf.filter(['ZSN_A', 'L_BLOOD', 'AGE', 'ALT_BLOOD', 'endocr_01',
                         'ROE', 'AST_BLOOD', 'lat_im', 'S_AD_ORIT', 'STENOK_AN', 'TIME_B_S', 
                         'D_AD_ORIT', 'INF_ANAM', 'n_r_ecg_p_06', 'zab_leg_01', 'ZSN'], axis = 1)

list(AllFeatdf.columns)

X1 = AllFeatdf.iloc[:, 0:15].values
y1 = AllFeatdf.iloc[:, 15].values

### Decision Tree with Select Features

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=1)

tree1 = DecisionTreeClassifier()
tree1.fit(X_train1,y_train1)
y_pred1 = tree1.predict(X_test1)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

param_grid1 = {'max_depth': [1, 2, 3, 4, 5, 6],
                     'criterion': ['gini', 'entropy'], 
                      'min_samples_leaf': [1,2,3,4,5,6],
                     'random_state' : [1,35,42],
                    'max_features' : ['sqrt', 'log2', None, 'auto']}

GridDT1 = GridSearchCV(estimator=tree1,
                  param_grid=param_grid1,
                  scoring='accuracy',
                  cv=10)
GridDT1.fit(X_train1,y_train1)

print("Best parameters found: ", GridDT1.best_params_) 
print("Best accuracy score found: %.3f "% GridDT1.best_score_)

from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(GridDT1, X_train1, y_train1,
                         scoring='accuracy', cv=10)

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

X1 = AllFeatdf.iloc[:, 0:15].values


y1 = AllFeatdf.iloc[:, 15].values

tree1 = DecisionTreeClassifier(criterion = 'entropy', max_depth= 2, 
                              max_features= None, min_samples_leaf= 1, 
                              random_state= 1)
tree1.fit(X_train1,y_train1)
y_pred1 = tree1.predict(X_test1)
confmat = confusion_matrix(y_true=y_test1, y_pred=y_pred1)
print(confmat)
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


precision = precision_score(y_test1, y_pred1, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test1, y_pred1, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test1, y_pred1, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))

p = sns.heatmap(confmat, annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Decision Tree Select Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('DT_select_CM.pdf')


### ROC of decision tree select features
from sklearn.metrics import roc_curve, auc
X1 = AllFeatdf.iloc[:, 0:15].values
y1 = AllFeatdf.iloc[:, 15].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train1)
X_test_std = sc.transform(X_test1)

tree1 = DecisionTreeClassifier(criterion = 'entropy', max_depth= 2, 
                              max_features= None, min_samples_leaf= 1, 
                              random_state= 1)
tree1.fit(X_train1,y_train1)

y_pred_proba = tree1.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test1, y_pred_proba)

#create ROC curve
plt.title('ROC of Decision Tree: Feature Selection', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()
plt.savefig('ROC_DT_select.pdf')


### Random Forest with select features

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

forest1 = RandomForestClassifier(n_estimators = 500, n_jobs = 2)
forest1.fit(X_train1, y_train1)

param_grid2 = {'max_depth': [1, 2, 3, 4, 5, 6],
                     'criterion': ['gini', 'entropy'], 
                      'min_samples_leaf': [1,2,3,4,5,6],
                     'random_state' : [1,35,42],
                    'max_features' : ['sqrt', 'log2', None]}

gs1 = GridSearchCV(estimator=forest1,
                  param_grid=param_grid2,
                  scoring='accuracy',
                  cv=10)
gs1.fit(X_train1, y_train1)
print("Best parameters found: ", gs1.best_params_) 
print("Best accuracy score found: %.3f "% gs1.best_score_)

from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(gs1, X_train1, y_train1,
                         scoring='accuracy', cv=10)

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

X1 = AllFeatdf.iloc[:, 0:15].values


y1 = AllFeatdf.iloc[:, 15].values

rf1 = RandomForestClassifier(criterion = 'gini', max_depth = 6,
                           random_state = 42, max_features = None,
                           min_samples_leaf=4, n_estimators=500, n_jobs=2)

rf1.fit(X_train1, y_train1)
y_pred1 = rf1.predict(X_test1)
confmat = confusion_matrix(y_true=y_test1, y_pred=y_pred1)
print(confmat)

p = sns.heatmap(confmat, annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Random Forest Select Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('RF_select_CM.pdf')

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


precision = precision_score(y_test1, y_pred1, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test1, y_pred1, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test1, y_pred1, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))

from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(rf1, X_train1, y_train1,
                         scoring='accuracy', cv=10)


### ROC of random forest with select features

import sklearn.metrics as metrics
X1 = AllFeatdf.iloc[:, 0:15].values
y1 = AllFeatdf.iloc[:, 15].values


rf1 = RandomForestClassifier(criterion = 'gini', max_depth = 6,
                           random_state = 42, max_features = None,
                           min_samples_leaf=4, n_estimators=500, n_jobs=2)

rf1.fit(X_train1, y_train1)
y_pred_proba = rf1.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test1, y_pred_proba)

#create ROC curve
plt.title('ROC of Random Forest: Feature Selection', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()


### SVM select features

X1 = AllFeatdf.iloc[:, 0:15].values
y1 = AllFeatdf.iloc[:, 15].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train1)
X_test_std = sc.transform(X_test1)



from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=1, C=0.0001, probability=True)
svm.fit(X_train_std, y_train1)

from sklearn.metrics import accuracy_score
y_pred = svm.predict(X_test_std)
print('Accuracy: %.4f' % accuracy_score(y_test1, y_pred))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test1, y_pred))


from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


precision = precision_score(y_test1, y_pred, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test1, y_pred, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test1, y_pred, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Select Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig("SVM_select_CM.pdf")


### ROC of SVM with select features

import sklearn.metrics as metrics
X1 = AllFeatdf.iloc[:, 0:15].values


y1 = AllFeatdf.iloc[:, 15].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train1)
X_test_std = sc.transform(X_test1)



from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=1, C=0.0001, probability=True)
svm.fit(X_train_std, y_train1)


y_pred_proba1 = svm.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_proba1)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test1, y_pred_proba1)

#create ROC curve
plt.title('ROC of SVM: Feature Selection', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()


### SVM with all features

X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=1, C=0.0001)
svm.fit(X_train_std, y_train)


from sklearn.metrics import classification_report, confusion_matrix
#print the summary of the number of correct and incorrect predictions
print(confusion_matrix(y_test, y_pred))

#print the performance evaluation metric
#Precision is defined as the ratio of true positives to the sum of true and false positives
#Recall is defined as the ratio of true positives to the sum of true positives and false negatives
#The F1 is the weighted harmonic mean of precision and recall. 
#The closer the value of the F1 score is to 1.0, the better the expected performance of the model is
#Support is the number of actual occurrences of the class in the dataset
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


precision = precision_score(y_test, y_pred, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test, y_pred, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test, y_pred, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM All Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('SVM_all_CM.pdf')


### SVM all features grid search

svm = SVC(random_state=1)
svm.fit(X_train_std, y_train)


param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range,
               'kernel': ['linear']},
              {'C': param_range,
               'gamma': param_range,
               'kernel': ['rbf']}]

gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,n_jobs=-1)


gs = gs.fit(X_train, y_train)
print(gs.best_params_)


### SVM with feature selection grid search

X1 = AllFeatdf.iloc[:, 0:15].values
y1 = AllFeatdf.iloc[:, 15].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train1)
X_test_std = sc.transform(X_test1)

svm = SVC(random_state=1)
svm.fit(X_train_std, y_train1)


param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]
param_grid = [{'C': param_range,
               'kernel': ['linear']},
              {'C': param_range,
               'gamma': param_range,
               'kernel': ['rbf']}]

gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,n_jobs=-1)


gs = gs.fit(X_train1, y_train1)
print(gs.best_params_)


### SVM ROC plot all features
X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=1, C=0.0001, probability=True)
svm.fit(X_train_std, y_train)


import sklearn.metrics as metrics


y_pred_proba = svm.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.title('ROC of SVM: All Features', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()


### Logistic Regression with all features

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

weights, params = [], []
for c in np.arange(-5, 5, dtype=float):
    logreg = LogisticRegression(C=5.**c, random_state=0, penalty = 'l2', max_iter = 1000)
    logreg.fit(X_train_std, y_train)
    weights.append(logreg.coef_[0])
    params.append(10**c)

weights = np.array(weights)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
#print the summary of the number of correct and incorrect predictions
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


precision = precision_score(y_test1, y_pred, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test1, y_pred, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test1, y_pred, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))

#print the performance evaluation metric
#Precision is defined as the ratio of true positives to the sum of true and false positives
#Recall is defined as the ratio of true positives to the sum of true positives and false negatives
#The F1 is the weighted harmonic mean of precision and recall. 
#The closer the value of the F1 score is to 1.0, the better the expected performance of the model is
#Support is the number of actual occurrences of the class in the dataset

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('LogReg All Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('LR_all_CM.pdf')


### ROC of Logistic Regression with all features
X = MIdf.iloc[:, 0:90].values
y = MIdf.iloc[:, 90].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

y_pred_proba = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.title('ROC of LogReg: All Features', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()


### Logistic Regression with feature selection
from sklearn.linear_model import LogisticRegression
X1 = AllFeatdf.iloc[:, 0:15].values
y1 = AllFeatdf.iloc[:, 15].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train1)
X_test_std = sc.transform(X_test1)

weights, params = [], []
for c in np.arange(-5, 5, dtype=float):
    logreg = LogisticRegression(C=5.**c, random_state=0, penalty = 'l2', max_iter = 1000)
    logreg.fit(X_train_std, y_train1)
    weights.append(logreg.coef_[0])
    params.append(10**c)

weights = np.array(weights)


from sklearn.metrics import classification_report, confusion_matrix
#print the summary of the number of correct and incorrect predictions
print(confusion_matrix(y_test1, y_pred))

#print the performance evaluation metric
#Precision is defined as the ratio of true positives to the sum of true and false positives
#Recall is defined as the ratio of true positives to the sum of true positives and false negatives
#The F1 is the weighted harmonic mean of precision and recall. 
#The closer the value of the F1 score is to 1.0, the better the expected performance of the model is
#Support is the number of actual occurrences of the class in the dataset
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


precision = precision_score(y_test1, y_pred, zero_division=0,average='weighted')
print('Precision score: {0:0.2f}'.format(precision))

recall = recall_score(y_test1, y_pred, zero_division=0,average='weighted')
print('Recall score: {0:0.2f}'.format(recall))

f1 = f1_score(y_test1, y_pred, zero_division=0,average='weighted')
print('f1 score: {0:0.2f}'.format(f1))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('LogReg Select Features Confusion Matrix', y=1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('LR_select_CM.pdf')


### ROC of Logistic Regression feature selection
y_pred_proba = logreg.predict_proba(X_test1)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test1, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
auc = metrics.roc_auc_score(y_test1, y_pred_proba)

#create ROC curve
plt.title('ROC of LogReg: Feature Selection', y=1)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.rcParams['figure.figsize'] = [10,10]
plt.show()
