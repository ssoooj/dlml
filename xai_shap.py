'''
SHAP is a game theoretic approach to explain the output of any machine learning model.
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory
and their related extensions.
'''

''' 

Standard Linear Regression

California housing dataset
: consists of 20,640 blocks of houses across California in 1990

Goal
: to predict the natural log of the median home price from 8 different features

1. MedInc - median income in block group
2. HouseAge - median house age in block group
3. AveRooms - average number of rooms per household
4. AveBedrms - average number of bedrooms per household
5. Population - block group population
6. AveOccup - average number of household members
7. Latitude - block group latitude
8. Longitude - block group longitude

'''
import pandas as pd
import shap
import sklearn

# A classic housing price dataset
X, y = shap.datasets.california(n_points = 1000)

# 100 instances for use as the background distribution
X100 = shap.utils.sample(X, 100)

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

'''
Examining the model coefficeints

-> The most common way of understanding a linear model is to examine the coefficients learned for each feature.

These coefficients tell us how much the model output changes when we change each of the input features
While coefficients are greate for telling us what will happen when we change the value of an input feature, 
by themselves they are not a great way to measure the overall importance of a feature.

This is because the value of each coefficient depends on the scale of the input features.
'''

print(f'Model coefficients : ')

for i in range(X.shape[1]):
	print(X.columns[i], '=', model.coef_[i].round(5))

'''
A more complete picture using partial dependence plots

: To understand a feature's importance in a model it is necessary to understand both
how changing that feature impacts the model's output, and also the distribution of that feature's values.
'''

shap.partial_dependence_plot(
	'MedInc', model.predict, X100, ice = False,
	model_expected_value = True, feature_expected_value = True)

# Compute the SHAP values for the linear model
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)

# Make a standard partial dependence plot
sample_ind = 20
shap.partial_dependence_plot(
	'MedInc', model.predict, X100, model_expected_value = True,
	feature_expected_value = True, ice = False,
	shap_values = shap_values[sample_ind:sample_ind + 1, :])

shap.plots.scatter(shap_values[:, 'MedInc'])

# The waterfall plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[sample_ind], max_display = 14)

# Explaining an additive regression model
# Fit a GAM model to the data
import interpret.glassbox

model_ebm = interpret.glassbox.ExplainableBoostingRegressor(interactions = 0)
model_ebm.fit(X, y)

# Explain the GAM model with SHAp
explainer_ebm = shap.Explainer(model_ebm.predict, X100)
shap_values_ebm = explainer_ebm(X)

# Make a standard partial dependence plot with a single SHAP value overlaid
fig, ax = shap.partial_dependence_plot(
	'MedInc', model_ebm.predict, X100, model_expected_value = True,
	feature_expected_value = True, show = False, ice = False,
	shap_values = shap_values_ebm[sample_ind:sample_ind + 1, :])

shap.plots.scatter(shap_values_ebm[:, 'MedInc'])

shap.plots.waterfall(shap_values_ebm[sample_ind])

shap.plots.beeswarm(shap_values_ebm)

# Explaining a non-additive boosted tree model
# Train XGBoost model
import xgboost

model_xgb = xgboost.XGBRegressor(n_estimators = 100, max_depth = 2).fit(X, y)

# Explain the GAM model with SHAP
explainer_xgb = shap.Explainer(model_xgb, X100)
shap_values_xgb = explainer_xgb(X)

# Make a standard partial dependence plot with a single SHAP value overlaid
fig, ax = shap.partial_dependence_plot(
	'MedInc', model_xgb.predict, X100, model_expected_value = True,
	feature_expected_value = True, show = False, ice = False, 
	shap_values = shap_values_xgb[sample_ind:sample_ind + 1, :])

shap.plots.scatter(shap_values_xgb[:, 'MedInc'], color = shap_values)

# Explaining a linear logistic regression model
# A classic adult census dataset price dataset
X_adult, y_adult = shap.datasets.adult()

# A simple linear logistic model
model_adult = sklearn.linear_model.LogisticRegression(max_iter = 10000)
model_adult.fit(X_adult, y_adult)

def model_adult_proba(x):
	return model_adult.predict_proba(x)[:, 1]

def model_adult_log_odds(x):
	p = model_adult.predict_log_proba(x)
	return p[:, 1] - p[:, 0]

# Make a standard partial dependence plot
sample_ind = 18

fig, ax = shap.partial_dependence_plot(
	'Capital Gain', model_adult_proba, X_adult, model_expected_value = True,
	feature_expected_value = True, show = False, ice = False)

# Compute the SHAP values for the linear model
background_adult = shap.maskers.Independent(X_adult, max_samples = 100)
explainer = shap.Explainer(model_adult_proba, background_adult)
shap_values_adult = explainer(X_adult[:1000])

shap.plots.scatter(shap_values_adult[:, 'Age'], color = shap_values)

# Compute the SHAP values for the linear model
explainer_log_odds = shap.Explainer(model_adult_log_odds, background_adult)
shap_values_adult_log_odds = explainer_log_odds(X_adult[:1000])

shap.plots.scatter(shap_values_adult_log_odds[:, 'Age'], color = shap_values)

# Make a standard partial dependence plot
sample_ind = 18

fig, ax = shap.partial_dependence_plot(
	'Age', model_adult_log_odds, X_adult, model_expected_value = True,
	feature_expected_value = True, show = False, ice = False)


# Train XGBoost model
model = xgboost.XGBClassifier(n_estimators = 100, max_depth = 2).fit(X_adult, y_adult*1, eval_metric = "logloss")

# Compute SHAP values
explainer = shap.Explainer(model, background_adult)
shap_values = explainer(X_adult)

# Set a display version of the data to use for plotting (has string values)
shap_values.display_data = shap.datasets.adult(display = True)[0].values

# By default a SHAP value bar plot will take the mean absolute value of each feature over all the instances of the dataset
shap.plots.bar(shap_values)

# Using the max absolute value highlights the Capital Gain and Capital Loss features
shap.plots.bar(shap_values.abx.max(0))

shap.plots.beeswarm(shap_values)

shap.plots.heatmap(shap_values[:1000])

shap.plots.scatter(shap_values[:, 'Age'], color = shap_values)

shap.plots.scatter(shap_values[:, 'Age'], color = shap_values[:, 'Capital Gain'])

shap.plots.scatter(shap_values[:, 'Relationship'], color = shap_values)

# Dealing with correlated features
clustering = shap.utils.hclust(X_adult, y_adult)

shap.plots.bar(shap_values, clustering = clustering, clustering_cutoff = 0.8)




