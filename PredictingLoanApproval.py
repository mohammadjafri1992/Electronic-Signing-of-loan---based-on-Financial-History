# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 01:18:55 2019

@author: Syed Jafri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Financial-Data.csv')

# Exploratory Data Analysis

dataset.head()
dataset.describe()
dataset.columns

# Data Cleaning

# Returns Boolean for each col if ANY data in the column is NULL
dataset.isna().any()

# Surprisingly, none of the cols have ANY missing values. This is not 
# a common occurance, but we can be certain that either;
# 1) The data source is very thorough in collecting the data, AND/OR
# 2) The data provider already cleaned the data for us to use.


# Building Histograms

dataset2 = dataset.drop(columns=['entry_id', 'pay_schedule', 'e_signed'])

# Here, we created another dataset to work on further data exploration.
# The data we have right now, i.e. the older dataset, contains some columns
# which are not very useful and will create unnecessary confusioin among 
# our plots. These columns are 'entry_id', 'pay_schedule', 'e_signed'.

# Since we have already done this following step for a TON of projects,
# so we are applying a popular coding concept called DRY Principles, i.e.
# Don't Repeat Yourself.
# Therefore, we are reusing this piece of code below which we have aleady
# coded a million times before.
# We need to change some variables depending on our requirements for
# this project, but this is going to be the only small modifications
# we are going to need here.

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

# The vals variable below is just a simple fail safe (or system crash safe)
# since this is going to be the bucket size of our histogram. This will prevent
# the system from creating total number of buckets greater than 100
# if the dataset has more than 100 unique values per column.

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# Now, if we remember the squence of steps we have preformed already
# for similar ML projects, then we would plot correlation with our 
# response variable

# Correlation barchart with Response Variable

# We are plotting a bar chart

dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with e-Signed", fontsize = 12,
        rot = 60, grid = True)


# Now, we are plotting our correlation matrix

sns.set(style='white')

# Computing correlation function
corr = dataset2.corr()

# Creating mask as we did in other projects

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18,15))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': .5})




































