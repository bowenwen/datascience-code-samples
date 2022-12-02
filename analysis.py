# %%

import pandas as pd
import numpy as np
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn import linear_model
from sklearn import tree
from sklearn.cluster import KMeans

# StandardScaler preferred over Normalizer, see note 1
# note 1: https://datascience.stackexchange.com/questions/45900/when-to-use-standard-scaler-and-when-normalizer


# %%
data_path = "../census-data/PUMF_Census_2016/dataverse_files/Data/Census_2016_Individual_PUMF.sav"
df = pd.read_spss(data_path)

# %%
# data processing of age_group
for age_group in list(df.AGEGRP.unique()):
    ag_str = age_group.replace(" years", "")
    if ag_str == "Not available":
        lower = -1
        upper = -1
    elif ag_str == "85 and over":
        lower = 85
        upper = 85
    else:
        lower = int(ag_str.split(" to ")[0])
        upper = int(ag_str.split(" to ")[1])
    df.loc[df.AGEGRP == age_group, "AGE"] = (lower + upper) / 2

# %%
print(list(df.columns))

# %%
sns.histplot(data=df, x="AGE")

# %%
# Analysis of wages
# LFACT == 1
df = df[(df.Wages < 88888888) & (df.LFACT == "Employed - Worked in reference week")]
df = df.reset_index(drop=True)

# %%
sns.boxplot(data=df, x="Sex", y="Wages")

# %%
sns.kdeplot(data=df, hue="Sex", x="Wages", bw_adjust=1.7)

# %%
sns.scatterplot(data=df, x="Wages", y="AGE")
sns.rugplot(data=df, x="Wages", y="AGE")

# %%
sns.kdeplot(data=df, x="Wages", hue="Sex")
sns.rugplot(data=df, x="Wages", hue="Sex")

# %%
df[df.Wages < 150000].plot.hexbin(x="Wages", y="AGE", gridsize=20)

# %%
wages_mobility = pd.pivot_table(
    df, index="Mob5", columns="MOB1", values="Wages", aggfunc="mean"
)
sns.heatmap(wages_mobility, annot=False, cmap="crest", vmin=30000, vmax=100000)

# %%
# recode variables
# Moved recently (1 year)
for cat in list(df.MOB1.unique()):
    if (cat == "Non-movers") or (cat == "Non-migrants"):
        df.loc[df.MOB1 == cat, "MOB1_cat"] = "not_moved_1"
    elif (
        (cat == "Interprovincial migrants")
        or (cat == "Different CSD, same census division")
        or (cat == "Different CD, same province")
    ):
        df.loc[df.MOB1 == cat, "MOB1_cat"] = "moved_within_canada_1"
    elif cat == "External migrants":
        df.loc[df.MOB1 == cat, "MOB1_cat"] = "moved_to_canada_1"
# Moved within 5 years
for cat in list(df.Mob5.unique()):
    if (cat == "Non-movers") or (cat == "Non-migrants"):
        df.loc[df.Mob5 == cat, "MOB5_cat"] = "not_moved_5"
    elif (
        (cat == "Interprovincial migrants")
        or (cat == "Different CSD, same census division")
        or (cat == "Different CD, same province")
    ):
        df.loc[df.Mob5 == cat, "MOB5_cat"] = "moved_within_canada_5"
    elif cat == "External migrants":
        df.loc[df.Mob5 == cat, "MOB5_cat"] = "moved_to_canada_5"
# Household size
for cat in list(df.HHSIZE.unique()):
    if cat == "Not available":
        df.loc[df.HHSIZE == cat, "HHSIZE_int"] = 1
    else:
        hh_size = cat.split(" ")[0]
        df.loc[df.HHSIZE == cat, "HHSIZE_int"] = hh_size
# Employment type
for cat in list(df.NOCS.unique()):
    nocs_cat = cat.split(" ")[0]
    # nocs_cat = cat[0:6]
    df.loc[df.NOCS == cat, "NOCS_cat"] = nocs_cat
# Combine employment categories
df["NOCS_cat"] = df.NOCS_cat.replace(["A", "C"], "mgmt_sci")
df["NOCS_cat"] = df.NOCS_cat.replace(["B"], "busi")
df["NOCS_cat"] = df.NOCS_cat.replace(["D", "E"], "health_edu_law")
df["NOCS_cat"] = df.NOCS_cat.replace(["F", "H", "J"], "art_trades_manuf")
df["NOCS_cat"] = df.NOCS_cat.replace(["G", "I"], "sales_agri")

# %%
# encode variables
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
lab_encode = LabelEncoder()
df["Sex_cat"] = lab_encode.fit_transform(df.Sex)
categories = list(lab_encode.classes_)
dict(zip(categories, lab_encode.transform(categories)))

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
cats = [
    [
        "No bedroom",
        "1 bedroom",
        "2 bedrooms",
        "3 bedrooms",
        "4 bedrooms",
        "5 bedrooms or more",
        "Not available",
    ]
]
ord_encode = OrdinalEncoder()
ord_encode = ord_encode.set_params(encoded_missing_value=-1, categories=cats)
df["BedRm_cat"] = ord_encode.fit_transform(df.BedRm.to_numpy().reshape(-1, 1))
dict(
    zip(cats[0], ord_encode.transform(np.array(cats).reshape(-1, 1)).reshape(1, -1)[0])
)

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe
# # try OneHotEncoder
# oht_encode = OneHotEncoder()
# oht_encode.fit(df['MOB5_cat'].unique().reshape(-1, 1))
# transformed = oht_encode.transform(
#     df['MOB5_cat'].to_numpy().reshape(-1, 1)).toarray()
# ohe_df = pd.DataFrame(transformed, columns=oht_encode.get_feature_names_out())
# df = pd.concat([df, ohe_df], axis=1).drop(['MOB5_cat'], axis=1)
# do the same thing with get dummies
df = pd.get_dummies(df, prefix=["MOB5_cat"], columns=["MOB5_cat"], drop_first=False)
df = pd.get_dummies(df, prefix=["MOB1_cat"], columns=["MOB1_cat"], drop_first=False)
df = pd.get_dummies(df, prefix=["NOCS_cat"], columns=["NOCS_cat"], drop_first=False)

# %%
print(list(df.columns))

# %% [markdown]
# Regression Models

# %%
# variable selection
# y - Wages
# X
# - MOB1_cat: moved in the last year
# - MOB5_cat: moved in the last 5 years
# "HHSIZE_int", "Sex_cat", "BedRm_cat", "MOB5_cat_moved_to_canada_5", "MOB5_cat_moved_within_canada_5", "MOB5_cat_not_moved_5", "MOB1_cat_moved_to_canada_1", "MOB1_cat_moved_within_canada_1", "MOB1_cat_not_moved_1"

# A1. simple regression model
pred_var = "Wages"
variable_list = [
    "AGE",
    "Sex_cat",
    "BedRm_cat",
    "MOB5_cat_moved_to_canada_5",
    "MOB5_cat_moved_within_canada_5",
    # "MOB5_cat_not_moved_5",
    # "MOB1_cat_moved_to_canada_1",
    # "MOB1_cat_moved_within_canada_1",
    "MOB1_cat_not_moved_1",
    "NOCS_cat_Not",
    "NOCS_cat_art_trades_manuf",
    # "NOCS_cat_busi",
    "NOCS_cat_health_edu_law",
    "NOCS_cat_mgmt_sci",
    "NOCS_cat_sales_agri",
]

X, y = df[variable_list], df[pred_var]

X = sm.add_constant(X)
variable_list.append("const")
model = sm.OLS(y, X).fit()
print(model.summary())

# %%

# A2. Simple regression model with formula (simple OLS)
# using formula is very convenient for specifying different models
# https://www.statsmodels.org/stable/api.html#statsmodels-formula-api
# print(' + '.join(variable_list))
results = smf.ols(
    "Wages ~ np.log(AGE) + Sex_cat + BedRm_cat + MOB5_cat_moved_to_canada_5 + MOB5_cat_moved_within_canada_5 + MOB1_cat_not_moved_1 + NOCS_cat_Not + NOCS_cat_art_trades_manuf + NOCS_cat_health_edu_law + NOCS_cat_mgmt_sci + NOCS_cat_sales_agri",
    data=df,
).fit()
print(results.summary())

# %%

# read: https://machinelearningresearch.quora.com/What-is-difference-between-ordinary-linear-regression-and-generalized-linear-model-or-linear-regression

# A3. Generalize Linear
results = smf.gls(
    "Wages ~ np.log(AGE) + Sex_cat + BedRm_cat + MOB5_cat_moved_to_canada_5 + MOB5_cat_moved_within_canada_5 + MOB1_cat_not_moved_1 + NOCS_cat_Not + NOCS_cat_art_trades_manuf + NOCS_cat_health_edu_law + NOCS_cat_mgmt_sci + NOCS_cat_sales_agri",
    data=df,
).fit()
print(results.summary())

# %% [markdown]
# # Model options from StatsModel
# - OLS: Ordinary Least Square - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
# - WLS: Weighted Least Square - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html#statsmodels.regression.linear_model.WLS
# - GLS: Generalized Least Square - https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html#statsmodels.regression.linear_model.GLS
# - GLSAR: Generalized Least Squares with AR covariance structure - specify covariance structure
# - MixedLM: Linear Mixed Effects - https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html#statsmodels.regression.mixed_linear_model.MixedLM
# - GEE, Ordinal GEE, Nominal GEE: Estimating equations / structural equations
# - RLM: Robust linear model - https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html#statsmodels.robust.robust_linear_model.RLM
# Discrete Choices - https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html
# - Logit: Logistic regression - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html#statsmodels.discrete.discrete_model.Logit
# - Probit: Probit regression - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html#statsmodels.discrete.discrete_model.Probit
# - MNLogit: Multinomial logit regression - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.MNLogit.html#statsmodels.discrete.discrete_model.MNLogit
# - Poisson: Poisson discrete choice - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Poisson.html#statsmodels.discrete.discrete_model.Poisson
# - Negative Binomial discrete choice - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html#statsmodels.discrete.discrete_model.NegativeBinomial
# Misc:
# - Quantile Regression - https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html#statsmodels.regression.quantile_regression.QuantReg
# - Hazard Regression - https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html#statsmodels.duration.hazard_regression.PHReg
# - GLM Additive Model - https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html#statsmodels.gam.generalized_additive_model.GLMGam

# feature selection
# - Variance Threshold
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
sel.fit(X)
print(sel.get_feature_names_out())


# %%
import math

# B1: ML Regression
# https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

# OLS Linear Regression using sklearn
reg = linear_model.LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Ridge Regression
reg = linear_model.Ridge(alpha=0.5)
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Lasso Regression
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Elastic Net with Cross Grid
reg = linear_model.ElasticNetCV(cv=5, random_state=0)
reg.fit(X, y)
y_pred = reg.predict(X)
print(reg.alpha_)
print(reg.intercept_, reg.coef_, reg.score(X, y))
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# # %%
# # Support Vector Machine for Regression
# from sklearn import svm

# reg = svm.SVR()
# reg.fit(X, y)
# y_pred = reg.predict(X)
# print(reg.alpha_)
# print(reg.intercept_, reg.coef_, reg.score(X, y))
# print(f"r2: {r2_score(y, y_pred)}")
# print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")

# %%
# Regression Tree
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

# normalize data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
transformer = Normalizer().fit(X)
Xt = transformer.transform(X)

reg = tree.DecisionTreeRegressor(max_depth=6)
reg.fit(Xt, y)
y_pred = reg.predict(Xt)
print(f"r2: {r2_score(y, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y, y_pred))}")
# tree.plot_tree(reg)
# https://stackoverflow.com/questions/68352933/name-of-variables-in-sklearn-pipeline
r = tree.export_text(reg, feature_names=variable_list)
print(r)
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# %%
# Random Forest
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = RandomForestClassifier(n_estimators=5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)


# %%
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = AdaBoostClassifier(n_estimators=5)
scores = cross_val_score(reg, X_train, y_train, cv=5)
print(f"scores: {scores}")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)


# %%
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(reg, X_train, y_train, cv=5)
print(f"scores: {scores}")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# %%
# Histogram-based Gradient Boosting
# https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
from sklearn.ensemble import HistGradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = HistGradientBoostingRegressor(random_state=0)
scores = cross_val_score(reg, X_train, y_train, cv=5)
print(f"scores: {scores}")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"r2: {r2_score(y_test, y_pred)}")
print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")
# pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# %%
# # Artificial Neural Network
# from sklearn.neural_network import MLPClassifier

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# reg = MLPClassifier(
#     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
# )
# scores = cross_val_score(reg, X_train, y_train, cv=5)
# print(f"scores: {scores}")
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# print(f"r2: {r2_score(y_test, y_pred)}")
# print(f"rmse: {math.sqrt(mean_squared_error(y_test, y_pred))}")

# # ~25 mins

# %% [markdown]
# Classification Models

# %%
# C1: ML Classification

pred_var = "BedRm"
variable_list = [
    "AGE",
    "Wages",
    "Sex_cat",
    "MOB5_cat_moved_to_canada_5",
    "MOB5_cat_moved_within_canada_5",
    # "MOB5_cat_not_moved_5",
    # "MOB1_cat_moved_to_canada_1",
    # "MOB1_cat_moved_within_canada_1",
    "MOB1_cat_not_moved_1",
    "NOCS_cat_Not",
    "NOCS_cat_art_trades_manuf",
    # "NOCS_cat_busi",
    "NOCS_cat_health_edu_law",
    "NOCS_cat_mgmt_sci",
    "NOCS_cat_sales_agri",
]

X, y = df[variable_list], df[pred_var]
y_classes = list(df[pred_var].unique())

# %%
# Classification Tree
# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# normalize data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
transformer = Normalizer().fit(X)
Xt = transformer.transform(X)

reg = tree.DecisionTreeClassifier(max_depth=6)
reg.fit(Xt, y)
y_pred = reg.predict(Xt)
print(f"score: {accuracy_score(y, y_pred)}")
print(classification_report(y, y_pred, target_names=y_classes))
# tree.plot_tree(reg)
# https://stackoverflow.com/questions/68352933/name-of-variables-in-sklearn-pipeline
r = tree.export_text(reg, feature_names=variable_list)
print(r)
pd.DataFrame({"name": list(X.columns), "value": reg.feature_importances_}).head(20)

# notes:
# Precision - TP / (TP + FP): fraction of true that is actually true
# Recall - TP / (TP + FN): fraction of true that was originally true
# f score - composite (avg) of precision and recall, 1 is perfect


# %% [markdown]
# Clustering Models

variable_list = [
    "AGE",
    "Wages",
    "Sex_cat",
    "MOB5_cat_moved_to_canada_5",
    "MOB5_cat_moved_within_canada_5",
    # "MOB5_cat_not_moved_5",
    # "MOB1_cat_moved_to_canada_1",
    # "MOB1_cat_moved_within_canada_1",
    "MOB1_cat_not_moved_1",
    "NOCS_cat_Not",
    "NOCS_cat_art_trades_manuf",
    # "NOCS_cat_busi",
    "NOCS_cat_health_edu_law",
    "NOCS_cat_mgmt_sci",
    "NOCS_cat_sales_agri",
]

X = df[variable_list]

# %%
# D1: ML Clustering

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
kmeans.labels_
y = kmeans.predict(X)
cc = kmeans.cluster_centers_
# dict(zip(list(range(0,len(cc))), cc))
cc_dict = {("cluster" + str(i)): cc[i] for i in range(0, len(cc))}
view_dict = {"name": variable_list}
view_dict.update(cc_dict)
pd.DataFrame(view_dict)

#%%
df.groupby('cluster').sum()['WEIGHT'].round(0)/df.sum()['WEIGHT']*100

# %%
df["cluster"] = y

# sns.scatterplot(data=df.sample(100), x="Wages", y="AGE", hue="cluster", size=10)

sns.displot(
    data=df.sample(100), x="Wages", y="AGE", hue="cluster", kind="kde", rug=True
)

# sns.displot(
#     data=df.sample(100), x="Wages", y="AGE", hue="cluster", kind="hist", rug=True
# )

pivot_df = df.pivot_table(index='cluster', columns='DPGRSUM', aggfunc='sum')['WEIGHT']
pivot_df / pivot_df.sum() * 100

# %% [markdown]
# Dimensionality Reduction Models

# %%
# E1: ML Dimensionality Reduction
