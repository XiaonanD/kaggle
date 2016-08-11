import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, pipeline, metrics, grid_search, cross_validation
%matplotlib inline

####Load train and test data¶
dtype_train = { #'AnimalID': np.str_,
         'Name': np.str_,
         'DateTime': np.str_,
         'AnimalType':np.str_,
         'SexuponOutcome':np.str_,
         'AgeuponOutcome':np.str_,
         'Breed': np.str_,
         'Color':np.str_,
         'OutcomeSubtype': np.str_,
         'OutcomeType':np.str_
    
}
#train is training set, labels are target value of training set
train = pd.read_csv('train.csv',dtype = dtype_train, usecols= dtype_train)
target = train['OutcomeType']
labels = train['OutcomeType']
train = train.drop(['OutcomeSubtype','OutcomeType'],axis =1)

# load test set
dtype_test = { 
         'Name': np.str_,
         'DateTime': np.str_,
         'AnimalType':np.str_,
         'SexuponOutcome':np.str_,
         'AgeuponOutcome':np.str_,
         'Breed': np.str_,
         'Color':np.str_,    
}
test = pd.read_csv('test.csv',dtype = dtype_test,usecols= dtype_test)

# Concatate training and test dataset together and call df_all
df_all = pd.concat((train,test), axis = 0, ignore_index= True)
df_all.shape
print(train.shape,test.shape)

#model_evaluation function and feature importance plot functions
from sklearn.grid_search import GridSearchCV
from time import time


#This is function to evalute models, scoring is 'log_loss'.
def model_evaluation(X_train, y_train, clf, param_grid, cv):
    model = GridSearchCV(estimator = clf,
                         param_grid = param_grid,
                         scoring = 'log_loss',
                         cv = cv
                         )
    #fit model
    model.fit(X_train,y_train)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters:", model.best_params_)
    return model

# function to plot feature_importance.
def plot_feature_importance(feature_importances,feature_names):
    ftr_imp_df = pd.DataFrame(sorted(zip(feature_names,feature_importances)
                          , key=lambda x: x[1], reverse = False)
                   )
    y_pos = np.arange(ftr_imp_df.shape[0])

    plt.barh(y_pos, ftr_imp_df[1], align='center', alpha=0.4)
    plt.yticks(y_pos, ftr_imp_df[0])
    plt.xlabel('Feature Importance')
    plt.show()

# LabelEncoder all features and target first as a benchmark
cat_columns = ['Name',
 'DateTime',
 'AnimalType',
 'SexuponOutcome',
 'AgeuponOutcome',
 'Breed',
 'Color']

#Since most of features are catergorical features,so firstly LabelEncoding all features.
LBL = preprocessing.LabelEncoder()

for col in cat_columns:
    print ("encoding %s"  %col)
    df_all[col] = LBL.fit_transform(df_all[col])

#Also Label encoding target. print target value and its corresponding assigned number.
LBL.fit(labels)

tgt_cls = dict(zip(labels.unique()
               , LBL.transform(labels.unique())))

print ("Target encoded as : ", tgt_cls)
labels = LBL.transform(labels)


######feature engineering: 1) datetime #####
## Get original 'DateTime' from train+test
a = pd.concat([train['DateTime'],test['DateTime']],axis =0,ignore_index= True)
df_all['DateTime'] = a

df_all['DateTime']=pd.to_datetime(df_all['DateTime'],infer_datetime_format = True, errors = 'coerce')

## get 7 new features, year, month, day, quater and weekday
df_all['year'] = df_all['DateTime'].dt.year
df_all['month'] = df_all['DateTime'].dt.month
df_all['day'] = df_all['DateTime'].dt.day
df_all['quarter'] = df_all['DateTime'].dt.quarter
df_all['weekday'] = df_all['DateTime'].dt.weekday
df_all['hour'] = df_all['DateTime'].dt.hour
df_all['weekOfYear'] = df_all['DateTime'].dt.weekofyear

##Remove the old feature
df_all = df_all.drop('DateTime',axis =1)
# quarter --1: Jan-Mar; 2:Apr-June; 3:July-Sept; 4:Oct-Dec
# weekday-- 0:Mon; 1:Tue...5:Saturday,6:Sunday

## get 7 new features, year, month, day, quater and weekday
df_all['year'] = df_all['DateTime'].dt.year
df_all['month'] = df_all['DateTime'].dt.month
df_all['day'] = df_all['DateTime'].dt.day
df_all['quarter'] = df_all['DateTime'].dt.quarter
df_all['weekday'] = df_all['DateTime'].dt.weekday
df_all['hour'] = df_all['DateTime'].dt.hour
df_all['weekOfYear'] = df_all['DateTime'].dt.weekofyear

##Remove the old feature
df_all = df_all.drop('DateTime',axis =1)
# quarter --1: Jan-Mar; 2:Apr-June; 3:July-Sept; 4:Oct-Dec
# weekday-- 0:Mon; 1:Tue...5:Saturday,6:Sunday

##feature engineering: 2) AgeuponOutcome################
# year/years, weeks, month/months, days/day, NaN
# Put AgeuponOutcome feature into numbers of years.
def cal_age_in_years(x):
    x = str(x)
    if x =='nan': 
        return 0
    age = int(x.split()[0])
    if 'year' in x:
        return age
    if 'month' in x:
        return age/12.
    if 'day' in x:
        return age/365.
    else: 
        return 0

###Get original 'AgeuponOutcome'
a = pd.concat([train['AgeuponOutcome'],test['AgeuponOutcome']],axis =0, ignore_index = True)
df_all['AgeuponOutcome'] = a

## Apply cal_age_in_years and change to number of years.
df_all['AgeuponOutcome']=df_all['AgeuponOutcome'].apply(cal_age_in_years)


#Add noise to Age *np.random.uniform(0.95,1.05), this is critical to avoid overfitting and make model robust.
mid_piv = train.shape[0]
df_all.AgeuponOutcome[:mid_piv] = df_all.AgeuponOutcome[:mid_piv]* np.random.uniform(0.95,1.05)    

#######Feature Engineering --3) Length of name    

a = pd.concat([train['Name'],test['Name']],axis =0, ignore_index = True)

df_all['LengthofName'] = a.apply(lambda row: len(str(row)))

##Feature Engineering 4) SexuponOutcome¶
###There are two information here: 1) Sex: female or male 2) Neutered/Spayed or intact
#Now try to separate these two pieces of information. And see if it can imporve the score
#---No, it doesn't help

#Feature Engineering 5) Breed
#This is a catergorical feature with high cardinality. I used "leave-one-out" method(Owen Zhang's method) 
#to engineer this feature. But from CV score and LB score, it doesn't help. 
#Finally I decided not to engineer this feature.


######Model fitting-tuning hyperparameters######
#Random forest as a base model, then extraTreesClassifier,XGboost
#History:
#1) 'max_features' :'sqrt' (the best)
#2) with verbose =5 (picking from 5,10)) it seems verbose is not very important, so just use default number

#2) By trying previously, the larger n_estimator, the better model(score) it is. 
# Therefore, directly use n_estimator = 2500


from sklearn.ensemble import RandomForestClassifier
model_rf2500 = RandomForestClassifier(n_estimators= 2500, random_state= 1234, criterion='gini')
#Plot feature importance in rf
feature_names = df_all.columns
plot_feature_importance(model.best_estimator_.feature_importances_,feature_names)

from sklearn.ensemble import ExtraTreesClassifier
model_extraTree = ExtraTreesClassifier(n_estimators= 1000, random_state= 1234, criterion='gini')

# After tune learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, n_estimator
#Below is the model and its parameters.

import xgboost as xgb
model_xgb= xgb.XGBClassifier(max_depth=9, learning_rate=0.01, n_estimators=1433, silent=False, 
                      objective='multi:softprob', nthread=-1, gamma=0.3, min_child_weight=3, subsample=0.9, 
                      colsample_bytree=0.5,  reg_lambda=1, seed=1234, missing=None)

###########Model Ensemble-blending###############
mid_piv = train.shape[0]
X = df_all[:mid_piv].values
y = labels.reshape(mid_piv)

from sklearn.cross_validation import StratifiedKFold

num_class = 5 # y has five different classes: Adoption, Transfer...
n_folds = 3

(train_x, train_y, test_x) = (X
                        ,y
                        ,df_all[mid_piv:].values)

skf = list(StratifiedKFold(train_y, n_folds))


clfs = [
        model_xgb,
        model_rf2500,
        model_extraTree
       ]

print "Creating train and test sets for blending."

train_blend_x = np.zeros((train_x.shape[0], len(clfs)*num_class))
test_blend_x = np.zeros((test_x.shape[0], len(clfs)*num_class))
scores = np.zeros ((len(skf),len(clfs)))

for j, clf in enumerate(clfs):
    print ("Blending model",j, clf)
    test_blend_x_j = np.zeros((test_x.shape[0], num_class))
    for i, (train, val) in enumerate(skf):
        print ("Model %d fold %d" %(j,i))
        train_x_fold = train_x[train]
        train_y_fold = train_y[train]
        val_x_fold = train_x[val]
        val_y_fold = train_y[val]
        clf.fit(train_x_fold, train_y_fold)
        val_y_predict_fold = clf.predict_proba(val_x_fold)
        score = metrics.log_loss(val_y_fold,val_y_predict_fold)
        print ("LOGLOSS: ", score)
        scores[i,j]=score
        train_blend_x[val, j*num_class:j*num_class+num_class] = val_y_predict_fold
        test_blend_x_j = test_blend_x_j + clf.predict_proba(test_x)
    test_blend_x[:,j*num_class:j*num_class+num_class] = test_blend_x_j/len(skf)
    print ("Score for model %d is %f" % (j,np.mean(scores[:,j])))

######Cross Validation/ Grid Search with blending
###Here for 2nd layer stacking, LogisticRegression is used.
# In this layer, the input is the ouput of first layer, and target is still the orginal target.
from sklearn.linear_model import LogisticRegression
LogisticRegression()
print  ("Blending.")
param_grid = {
              }
model = model_evaluation(train_blend_x
                                         , train_y
                                         , LogisticRegression()
                                         , param_grid
                                         , cv=5
                                         )   

print ("best params:", model.best_params_)


######Prediction and submission#######
prediction = model.predict_proba(test_blend_x)
prediction.shape
submission = pd.read_csv('sample_submission.csv')
submission[['Adoption','Died','Euthanasia','Return_to_owner','Transfer']] = prediction
submission.to_csv('Final-blending.csv',index = False)
print ('job done. csv ready in your expedia folder.')


