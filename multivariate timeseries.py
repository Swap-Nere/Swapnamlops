#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[10]:


import json

import io
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer


import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose


# In[11]:


df=pd.read_html("sp500-vs-dow-jones-vs-go.xls")


# In[12]:


df


# In[13]:


df_main=df[0]


# In[14]:


df_main.columns


# In[15]:


df_main.info()


# In[16]:


dft=df_main.fillna(0)


# In[17]:


dft


# In[18]:


dft.isnull().sum()


# In[19]:


dft['DateTime']=pd.to_datetime(dft['DateTime'])


# In[20]:


dft.info()


# In[21]:


dft


# In[22]:


dft = dft.set_index("DateTime")


# In[23]:


dft


# In[24]:


import matplotlib.pyplot as plt
 
# Using a inbuilt style to change
# the look and feel of the plot
plt.style.use("fivethirtyeight")
 
# setting figure size to 12, 10
plt.figure(figsize=(12, 10))
 
# Labelling the axes and setting
# a title
plt.xlabel("DateTime")
plt.ylabel("values")
plt.title(" Time Series Plot")
 
# plotting the "A" column alone
plt.plot(dft['S&P 500'])


# In[25]:


plt.style.use("fivethirtyeight")
dft.plot(subplots=True, figsize=(12, 15))


# In[26]:


# import matplotlib.pyplot as plt
 
# # Using a inbuilt style to change
# # the look and feel of the plot
# plt.style.use("fivethirtyeight")
 
# # setting figure size to 12, 10
# plt.figure(figsize=(25, 10))
 
# # Labelling the axes and setting a
# # title
# plt.xlabel("DateTime")
# plt.ylabel("Values")
# plt.title("Bar Plot of 'S&P 500'")

 
# # plotting the "A" column alone
# plt.bar(dft.index, dft["S&P 500"], width=5)


# In[27]:


dft.corr()


# In[28]:


dft['S&P 500'].corr(dft['Dow Jones'])


# In[29]:


dft['S&P 500'].corr(dft['Gold'])


# In[30]:


dft['S&P 500'].corr(dft['Silver'])


# In[31]:


dft['Dow Jones'].corr(dft['Silver'])


# In[32]:


dft['Dow Jones'].corr(dft['Gold'])


# In[33]:


dft['Dow Jones'].corr(dft['S&P 500'])


# In[34]:


dft['Gold'].corr(dft['S&P 500'])


# In[35]:


dft['Gold'].corr(dft['Silver'])


# In[36]:


dft['Gold'].corr(dft['Dow Jones'])


# In[37]:


dft['Silver'].corr(dft['Dow Jones'])


# In[38]:


dft['Silver'].corr(dft['Gold'])


# In[39]:


dft['Silver'].corr(dft['S&P 500'])


# In[40]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
x=dft['Silver']
y=dft['Gold']
plt.scatter (x,y)

plt.show()


# In[41]:


corrmat = dft.corr()
  
f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)


# In[42]:


output=dft.to_json('temp.json')


# In[43]:


output


# In[44]:


dft.describe()


# In[45]:


get_ipython().system('pip install pmdarima')


# In[46]:


import pandas as pd
import numpy as np


# In[47]:


df= pd.read_html("sp500-vs-dow-jones-vs-go.xls",index_col="DateTime",parse_dates=True)


# In[48]:


df


# In[49]:


df_main=df[0]


# In[50]:


dft=df_main.dropna()


# In[51]:


dft


# In[52]:


dft.shape


# In[53]:


dft.head()


# In[54]:


dft['S&P 500'].plot(figsize=(12,5))


# In[55]:


dft['Dow Jones'].plot(figsize=(12,5))


# In[56]:


dft['Gold'].plot(figsize=(12,5))


# In[57]:


dft['Silver'].plot(figsize=(12,5))


# In[58]:


#VAR Model

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[59]:


df= pd.read_html("sp500-vs-dow-jones-vs-go.xls",index_col="DateTime",parse_dates=True)


# In[60]:


df


# In[61]:


df_main=df[0]


# In[62]:


dft=df_main.dropna()


# In[63]:


dft


# In[64]:


dft.shape


# In[65]:


dft.head()


# In[66]:


dftt=dft.dropna()


# In[67]:


fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = dftt[dftt.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(dftt.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=2)

plt.tight_layout();


# In[68]:


ad_fuller_result_1 = adfuller(dftt['S&P 500'].diff()[1:])

print('realS&P 500')
print(f'ADF Statistic: {ad_fuller_result_1[0]}')
print(f'p-value: {ad_fuller_result_1[1]}')

print('\n---------------------\n')

ad_fuller_result_2 = adfuller(dftt['Dow Jones'].diff()[1:])

print('realDow Jones')
print(f'ADF Statistic: {ad_fuller_result_2[0]}')
print(f'p-value: {ad_fuller_result_2[1]}')
ad_fuller_result_2 = adfuller(dftt['Dow Jones'].diff()[1:])

ad_fuller_result_3 = adfuller(dftt['Silver'].diff()[1:])

print('realSilver')
print(f'ADF Statistic: {ad_fuller_result_2[0]}')
print(f'p-value: {ad_fuller_result_2[1]}')
ad_fuller_result_3 = adfuller(dftt['Silver'].diff()[1:])

ad_fuller_result_4 = adfuller(dftt['Gold'].diff()[1:])

print('realGold')
print(f'ADF Statistic: {ad_fuller_result_2[0]}')
print(f'p-value: {ad_fuller_result_2[1]}')
ad_fuller_result_4 = adfuller(dftt['Gold'].diff()[1:])


# In[69]:


print('S&P 500 causes Dow Jones\n')
print('------------------')
granger_1 = grangercausalitytests(dftt[['Dow Jones', 'S&P 500']], 4)

print('Dow Jones causes S&P 500?\n')
print('------------------')
granger_2 = grangercausalitytests(dftt[['S&P 500', 'Dow Jones']], 4)


# In[70]:


print('Gold causes Silver?\n')
print('------------------')
granger_1 = grangercausalitytests(dftt[['Silver', 'Gold']], 4)

print('\Silver causes Gold?\n')
print('------------------')
granger_2 = grangercausalitytests(dftt[['Gold', 'Silver']], 4)


# In[71]:


dftt = dftt[['S&P 500','Dow Jones','Gold','Silver']]
print(dftt.shape)


# In[72]:


train_df=dftt[:-12]
test_df=dftt[-12:]


# In[73]:


print(test_df.shape)


# In[74]:


model = VAR(train_df.diff()[1:])


# In[75]:


sorted_order=model.select_order(maxlags=20)
print(sorted_order.summary())


# In[76]:


var_model = VARMAX(train_df, order=(4,0),enforce_stationarity= True)
fitted_model = var_model.fit(disp=False)
print(fitted_model.summary())


# In[77]:


n_forecast = 12
predict = fitted_model.get_prediction(start=len(train_df),end=len(train_df) + n_forecast-1)#start="1989-07-01",end='1999-01-01')

predictions=predict.predicted_mean


# In[78]:


predictions.columns=['S&P 500_predicted','Dow Jones_predicted','Gold_predicted','Silver_predicted']
predictions


# In[79]:


test_vs_pred=pd.concat([test_df,predictions],axis=1)


# In[80]:


test_vs_pred.plot(figsize=(12,5))


# In[81]:


from sklearn.metrics import mean_squared_error
import math 
from statistics import mean

rmse_SP =math.sqrt(mean_squared_error(predictions['S&P 500_predicted'],test_df['S&P 500']))
print('Mean value of S&P 500 is : {}. Root Mean Squared Error is :{}'.format(mean(test_df['S&P 500']),rmse_SP))

rmse_DJ=math.sqrt(mean_squared_error(predictions['Dow Jones_predicted'],test_df['Dow Jones']))
print('Mean value of DowJOnes is : {}. Root Mean Squared Error is :{}'.format(mean(test_df['Dow Jones']),rmse_DJ))

rmse_Gold=math.sqrt(mean_squared_error(predictions['Gold_predicted'],test_df['Gold']))
print('Mean value of Gold is : {}. Root Mean Squared Error is :{}'.format(mean(test_df['Gold']),rmse_Gold))

rmse_Silver=math.sqrt(mean_squared_error(predictions['Silver_predicted'],test_df['Silver']))
print('Mean value of Silver is : {}. Root Mean Squared Error is :{}'.format(mean(test_df['Silver']),rmse_Silver))


# In[89]:


with open('metrics.txt','w')as outputfile:
     outputfile.write(f'\nRoot Mean Squared Error ={rmse_SP,rmse_DJ,rmse_Gold,rmse_Silver}.')
    


# In[93]:


outputfile


# In[ ]:




