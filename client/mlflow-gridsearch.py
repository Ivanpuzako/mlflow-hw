import numpy as np
import pandas as pd 
import argparse
import os
import mlflow
from sklearn.preprocessing import scale
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
from sklearn.metrics import make_scorer
warnings.filterwarnings('ignore')

data_path = 'data.csv'
df = pd.read_csv(data_path, index_col='PassengerId')

df = df.drop(columns=['Cabin'])
df['Age'] = df['Age'].fillna(df.groupby(['Pclass'])['Age'].transform(np.mean))

df['Embarked'] = df['Embarked'].fillna('S')

sex_replace = {'female': 0, 'male': 1}
df = df.replace({'Sex': sex_replace})

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Parch']]
y = df[['Survived']]
X_scaled = scale(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=101)

parameters = {'kernel': ['linear', 'sigmoid'], 'C':[0.5,1,2]}
score = {'AUC': 'roc_auc', 'precision': make_scorer(metrics.precision_score), 'recall':make_scorer(metrics.recall_score)}

gridsearch = GridSearchCV(SVC(), parameters, score, refit = "AUC", cv=3)
gridsearch.fit(X_train, y_train)

cv_results = gridsearch.cv_results_

model_name = "SVM"
num_params = len(cv_results['params'])
mlflow.set_tracking_uri("http://server:5000")
for run_index in range(num_params):
   with mlflow.start_run(run_name=str(run_index)) as run:  

      mlflow.log_param("folds", gridsearch.cv)

      print("Logging parameters")
      params = list(gridsearch.param_grid.keys())
      for param in params:
         mlflow.log_param(param, cv_results["param_%s" % param][run_index])

      print("Logging metrics")
      for score_name in [score for score in cv_results if "mean_test" in score]:
         mlflow.log_metric(score_name, cv_results[score_name][run_index])
        
      if cv_results['params'][run_index] == gridsearch.best_params_:
         mlflow.set_tag("model", 'Saved')
         print("Logging model")  
         mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name)
      else:
         mlflow.set_tag("model", 'nothing')