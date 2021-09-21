#!/usr/bin/env python
# coding: utf-8

# In[41]:


#import data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/tahir/Documents/Python Datasets/medical examination.csv")
df.head()


# In[9]:


#add overweight column

ibm = (df["weight"] / ((df["height"]/100) ** 2 ))
ibm = np.array(ibm)
print(ibm)

ibm = np.where(ibm > 25,1,0)
df['overweight'] = ibm.astype(int)
print(df["overweight"])


# In[12]:


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df["cholesterol"] = np.where((df["cholesterol"] == 1), 0, 1)
df["gluc"] = np.where((df["gluc"]) == 1, 0, 1)


# In[13]:


print(df["cholesterol"])


# In[14]:


df_cat = pd.melt(df, id_vars = ["cardio"], value_vars = ['active','alco',"cholesterol",'gluc','overweight','smoke'])


# In[15]:


print(df_cat)


# In[16]:


df_cat = df_cat.groupby(["cardio"]).head()


# In[24]:


figure = sns.catplot(x = "variable", kind = "count", hue = "value", data = df_cat, col = "cardio")
figure.set_axis_labels("variable","total")


# In[20]:


figure.set_axis_labels("variable","total")


# In[25]:


fig = figure.fig
fig.savefig('catplot.png')


# In[33]:


print(df['ap_lo']) 


# In[34]:


df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025))

& (df['height'] <= df['height'].quantile(0.975) )

& (df['weight'] >= df['weight'].quantile(0.025))

& (df['weight'] <= df['weight'].quantile(0.975) ) ]


# In[35]:


corr = df_heat.corr()


# In[36]:


print(corr)


# In[38]:


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype= bool))


# In[42]:


# Set up the matplotlib figure
fig, ax = plt.subplots(figsize = (10,10))


# In[46]:


dfplot = sns.heatmap(corr, fmt=".1f", vmax= 0.26,annot_kws={'size':10}, cmap='PRGn', annot=True, mask=mask)


# In[47]:


plt.show()


# In[ ]:




