# Importing necessary libraries and loading the dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('appdata10.csv')

# Basic exploration and preprocessing
# Includes viewing data, describing numerical variables, and extracting hour from timestamp
dataset.head(10)
dataset.describe()
dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)

# Preparing data for visualization by dropping unnecessary columns
dataset2 = dataset.copy().drop(columns=['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
dataset2.head()

# Generating histograms for numerical columns to understand distributions
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Analyzing correlations to identify potential predictors
# Includes correlation with response variable and correlation matrix among features
dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10), title='Correlation with Response variable', fontsize=15, rot=45, grid=True)

sn.set(style="white", font_scale=2)
corr = dataset2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize=40)
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
