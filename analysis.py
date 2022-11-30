# %%
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns

# %%
data_path = "../census-data/PUMF_Census_2016/dataverse_files/Data/Census_2016_Individual_PUMF.sav"
df = pd.read_spss(data_path)

# %%
# data processing
for age_group in list(df.AGEGRP.unique()):
    ag_str = age_group.replace(' years', '')
    if ag_str == 'Not available':
        lower = -1
        upper = -1
    elif ag_str == '85 and over':
        lower = 85
        upper = 85
    else:
        lower = int(ag_str.split(' to ')[0])
        upper = int(ag_str.split(' to ')[1])
    df.loc[df.AGEGRP == age_group, 'AGE'] = (lower + upper) / 2


# %%
print(list(df.columns))

# %%
sns.histplot(data=df, x='AGE')

# %%
# Analysis of wages
# LFACT == 1
df = df[(df.Wages < 88888888) & (
    df.LFACT == 'Employed - Worked in reference week')]

# %%
sns.boxplot(data=df, x='Sex', y='Wages')

# %%
sns.kdeplot(data=df, hue='Sex', x='Wages', bw_adjust=1.7)

# %%
sns.scatterplot(data=df, x='Wages', y='AGE')
sns.rugplot(data=df, x='Wages', y='AGE')

# %%
sns.kdeplot(data=df, x='Wages', hue='Sex')
sns.rugplot(data=df, x='Wages', hue='Sex')

# %%
df[df.Wages < 150000].plot.hexbin(
    x='Wages', y='AGE', gridsize=20)

# %%
wages_mobility = pd.pivot_table(
    df, index="Mob5", columns='MOB1', values='Wages', aggfunc='mean')
sns.heatmap(wages_mobility, annot=False, cmap="crest", vmin=30000, vmax=100000)


# %%
# make a simple regression model
lab_encode = LabelEncoder()
# df['MOB1'].isna().sum()
df['MOB1_cat'] = lab_encode.fit_transform(df.MOB1)
df['MOB5_cat'] = lab_encode.fit_transform(df.Mob5)
# df[['MOB1', 'MOB1_cat']]

reg = linear_model.LinearRegression()
X, y = df[['MOB1_cat', 'MOB5_cat']], df.Wages
reg.fit(X, y)
print(reg.intercept_, reg.coef_, reg.score(X, y))

# %%
import statsmodels.api as sm
model = sm.OLS(y, X).fit()
print(model.summary())

# %%
