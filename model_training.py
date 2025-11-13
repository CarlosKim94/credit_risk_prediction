import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# read the data
df = pd.read_csv("data/credit_risk_dataset.csv")

# replace missing values with 0.0
df['person_emp_length'] = df['person_emp_length'].fillna(0.0)
df['loan_int_rate'] = df['loan_int_rate'].fillna(0.0)

# split data into training, validation, and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.loan_status.values
y_val = df_val.loan_status.values
y_test = df_test.loan_status.values
del df_train['loan_status']
del df_val['loan_status']
del df_test['loan_status']

# Random Forest model training
def rf_train(df_train, y_train, n_estimators=10, max_depth=10):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)

    return dv, rf

# Random Forest model prediction
def rf_predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

# set hyper parameter range
estimators = np.arange(10, 210, 10)
depth = [10, 15, 20, 25]

auc_scores = []
best_score = 0

# train the model and save the best performing model
for d in depth:
    for n in estimators:
        dv, model = rf_train(df_train, y_train, n_estimators=n, max_depth=d)
        y_pred = rf_predict(df_val, dv, model)
        val_roc_auc = round(roc_auc_score(y_val, y_pred),3)
        auc_scores.append((d, n, val_roc_auc))

        if val_roc_auc > best_score:
            best_score = val_roc_auc
            dir = 'model'
            model_checkpoint = f'model_depth_{d}_estimator_{n}_{val_roc_auc}.bin'
            checkpoint = os.path.join(dir, model_checkpoint)
            with open(checkpoint, 'wb') as f_out:
                pickle.dump((dv, model), f_out)
            print(f"Check point saved: {checkpoint}")