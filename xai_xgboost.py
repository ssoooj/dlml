import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
from xgboost import plot_tree, plot_importance
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Read data

df = pd.read_csv('/Users/sohyeon/Downloads/diabetes.csv')
vals = df.values

# Dataset from Numpy to DataFrame

X_features = vals[:, 0:8]
y_label = vals[:, 8]

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
target = ['Normal', 'Diabetes']
# print("columns : ", dataset.columns, "\n", "shape : ", dataset.shape)

diabetes_df = pd.DataFrame(data = X_features, columns = cols)
diabetes_df['target'] = y_label

X_features = diabetes_df.iloc[:, 0:8]
y_label = diabetes_df.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X_features, y_label, 
									test_size = 0.2, random_state = 7)
print(x_train.shape, x_test.shape)

# XGB model

xgb_model = XGBClassifier(n_estimator = 1000, 
	learning_rate = 0.001, 
	max_depth = 5)

evals = [(x_test, y_test)]

xgb_model.fit(x_train, y_train, 
	early_stopping_rounds = 100, 
	eval_metric = 'logloss', 
	eval_set = evals, 
	verbose = False)

y_pred = xgb_model.predict(x_test)
predictions = [round(value) for value in y_pred]
# print(f'Predictions : {predictions}')

# LightGBM model
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(n_estimators = 1000, 
	learning_rate = 0.001,
	num_leaves = 30, 
	n_jobs = -1, 
	boost_from_average = True)
lgb_model.fit(x_train, y_train)

# Evaluation

accuracy = accuracy_score(y_test, predictions)
print(f'\nAccuracy : {accuracy * 100.0:.2f}\n')

expected_y = y_test
predicted_y = xgb_model.predict(x_test)

report = metrics.classification_report(expected_y, predicted_y, target_names = target)
conf_matrix = metrics.confusion_matrix(expected_y, predicted_y)
print(f'XGB report : {report}\nXGB confusion matrix :\n{conf_matrix}\n')

lgb_predicted_y = lgb_model.predict(x_test)

lgb_accuracy = accuracy_score(y_test, lgb_predicted_y)
print(f'LightGBM model accuracy : {lgb_accuracy * 100.0:.2f}\n')

report2 = metrics.classification_report(expected_y, lgb_predicted_y, target_names = target)
cm = confusion_matrix(y_test, lgb_predicted_y)
print(f'LightGBM report : {report2}\nLightGBM confusion matrix\n{cm}')
# print('\nTrue Positives(TP) = ', cm[0,0])
# print('\nTrue Negatives(TN) = ', cm[1,1])
# print('\nFalse Positives(FP) = ', cm[0,1])
# print('\nFalse Negatives(FN) = ', cm[1,0])

# Diagnosis with new data using xgb model

value = np.array([0, 180, 75, 45, 0, 18.2, 0.353, 30], ndmin = 2)
value_df = pd.DataFrame(data = value, columns = cols)
print(value_df)

l = xgb_model.predict_proba(value_df)
print(f'\nXGB Diagnosis Test\nNormal : {l[0][0]:.2f} Diabetes : {l[0][1]:.2f}')

l2 = lgb_model.predict_proba(value_df)
print(f'\nLGB Diagnosis Test\nNormal : {l2[0][0]:.2f} Diabetes : {l2[0][1]:.2f}')
# l = model.predict_proba(x_test)[:, 0]

# Visualization

rcParams['figure.figsize'] = 20, 20

plot_importance(xgb_model)
plt.savefig('/Users/sohyeon/Downloads/feature importance.png', dpi = 300)

plot_tree(xgb_model, filled = True)
plt.savefig('/Users/sohyeon/Downloads/tree_model.png', dpi = 300)

matrix = plot_confusion_matrix(xgb_model, x_test, y_test, cmap = plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color = 'white')
plt.xlabel('Predicted Label', color = 'black')
plt.ylabel('True Label', color = 'black')
plt.gcf().axes[0].tick_params(colors = 'black')
plt.gcf().axes[1].tick_params(colors = 'black')
plt.savefig('/Users/sohyeon/Downloads/xgb_confusion_matrix.png', dpi = 300)

matrix2 = plot_confusion_matrix(lgb_model, x_test, y_test, cmap = plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color = 'white')
plt.xlabel('Predicted Label', color = 'black')
plt.ylabel('True Label', color = 'black')
plt.gcf().axes[0].tick_params(colors = 'black')
plt.gcf().axes[1].tick_params(colors = 'black')
plt.savefig('/Users/sohyeon/Downloads/lgb_confusion_matrix.png', dpi = 300)
