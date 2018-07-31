#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 07:10:53 2018

@author: craig
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, auc, roc_curve, classification_report

# =============================================================================
# Prepare the data
# =============================================================================
df = pd.read_csv('./csv/diabetic_model2.csv')

features = [col for col in df.columns if col != 'single_condition']

X, y = df[features], df['single_condition']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33,
                                                    random_state=23)

# =============================================================================
# Setup the model
# =============================================================================
ss = StandardScaler()

logger = linear_model.LogisticRegression()

log_pipe = Pipeline([
    ('ss', ss),
    ('logger', logger)
])

params = {
    'logger__penalty': ['l1', 'l2'],
    'logger__C': np.logspace(0, 2.5, num=16, base=np.e)
}
gs = GridSearchCV(estimator=log_pipe, param_grid=params, n_jobs=4)
gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

# =============================================================================
# Build the model
# =============================================================================
lrmod = linear_model.LogisticRegression(
            penalty=gs.best_params_['logger__penalty'],
            C=gs.best_params_['logger__C']
            )

Xs_train = ss.fit_transform(X_train)
Xs_test = ss.transform(X_test)

lrmod.fit(Xs_train, y_train)

# =============================================================================
# Predict
# =============================================================================
y_hat = lrmod.predict(Xs_test)

# =============================================================================
# Baseline Accuracy # = 0.6231592426598372
# =============================================================================
y.value_counts().max() / len(y) 

# =============================================================================
# Evaluate the Model
# =============================================================================
cm = confusion_matrix(y_test, y_hat)

sns.set_context(context='notebook', font_scale=1.5)
ax = plt.subplot()
sns.heatmap(cm, annot=True, cmap='RdBu', cbar=False, fmt='g')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Single', 'Compound'])
ax.yaxis.set_ticklabels(['Compound', 'Single'])
plt.show()

print(classification_report(y_test, y_hat))

# =============================================================================
# Save the model in the cwd
# =============================================================================
pkl_filename = 'diabetic_logreg.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(lrmod, file)