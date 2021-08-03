#!/usr/bin/env python
# coding: utf-8

# # Bike-Sharing Demand Analysis

# ## Import packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## 1.Load the data

# In[2]:


data = pd.read_csv("hour.csv")


# In[3]:


data.head(2)


# In[4]:


data.rename(columns = {"dteday":"date","mnth":"month"}, inplace = True )


# ## 2. Check for null values in the data and drop records with NAs.

# In[5]:


data.isna().sum()


# There are no null records.

# ## 3.Sanity checks:

# ### 3.1 Check if registered + casual = cnt for all the records. If not, the row is  junk and should be dropped.

# In[6]:


np.sum((data.casual + data.registered) != data.cnt)


# In[7]:


#OR


# In[8]:


data[(data.casual + data.registered) != data.cnt]


# ### 3.2 Month values should be 1-12 only

# In[9]:


np.unique(data.month)


# In[10]:


#OR


# In[11]:


data[data.month>12]


# ### 3.3 Hour values should be 0-23 

# In[12]:


np.unique(data.hr)


# In[13]:


#OR


# In[14]:


data[data.hr > 23]


# ## 4.The variables ‘casual’ and ‘registered’ are redundant and need to be dropped. ‘Instant’ is the index and needs to be dropped too. The date column dteday will not be used in the model building, and therefore needs to be dropped. Create a new dataframe named inp1.

# In[15]:


inp1 = data.drop(["casual","registered","instant","date"],axis =1)


# In[16]:


inp1.head(2)


# ## 5. Univariate analysis: 

# ### 5.1 Describe the numerical fields in the dataset using pandas describe method.

# In[17]:


inp1.describe()


# ### 5.2 Make density plot for temp. This would give a sense of the centrality and the spread of the distribution.

# In[18]:


sns.distplot(inp1.temp, hist = False)
plt.show()


# In[19]:


##OR


# In[20]:


inp1.temp.plot.density()
plt.xlabel("temp")
plt.show()


# ### 5.3 Boxplot for atemp .Are there any outliers?

# In[21]:


sns.boxplot(x= inp1.atemp)
plt.show()


# There are no outliers present in atemp.

# ### 5.4 Histogram for hum.Do you detect any abnormally high values? 

# In[22]:


sns.distplot(inp1.hum, kde = False)
plt.show()


# In[23]:


##OR


# In[24]:


plt.hist(inp1.hum)
plt.xlabel("hum")
plt.show()


# There are no visible abnormally high values.

# ### 5.5 Density plot for windspeed

# In[25]:


inp1.windspeed.plot.density()
plt.show()


# In[26]:


##OR


# In[27]:


sns.distplot(inp1.windspeed, hist = False)
plt.show()


# ### 5.6 Box and density plot for cnt – this is the variable of interest .Do you see any outliers in the boxplot? Does the density plot provide a similar insight? 

# In[28]:


sns.boxplot(x=inp1.cnt)
plt.show()


# There are outliers present in cnt.

# In[29]:


sns.distplot(inp1.cnt, hist = False)
plt.show()


# Both Boxplot and Density plot shows the similar picture - there are high values in count

# ## 6. Outlier treatment:  

# ### 6.1 Cnt looks like some hours have rather high values. You’ll need to treat these outliers so that they don’t skew the analysis and the model. 
# 6.1.1Find out the following percentiles: 10, 25, 50, 75, 90, 95, 99
# 
# 6.1.2Decide the cutoff percentile and drop records with values higher than the cutoff. Name the new dataframe as inp2. 

# In[30]:


inp1.cnt.quantile([0.10,0.25,0.50,0.75,0.90,0.95,0.99])


# Taking 95 as cutoff percentile, we will drop the values higher than this

# In[31]:


inp2= inp1.copy()


# In[32]:


inp2 = inp2[inp2.cnt<563]


# ## 7. Bivariate Analysis: 

# ### 7.1 Make boxplot for cnt vs. hour.What kind of pattern do you see? 

# In[33]:


plt.figure(figsize = (12,6))
sns.boxplot(x="hr", y="cnt", data = inp2)
plt.show()


# It can be said that during 7-8am demand of bike is high, and than later in the evening demand of bike again increases, mostly between 17-18.

# ### 7.2 Make boxplot for cnt vs. weekday.Is there any difference in the rides by days of the week? 

# In[34]:


plt.figure(figsize = (10,6))
sns.boxplot(x="weekday", y="cnt", data = inp2)
plt.show()


# There is not much difference in the rides by day of the weeks

# ### 7.3  Make boxplot for cnt vs. month.Look at the median values. Any month(s) that stand out?

# In[35]:


plt.figure(figsize = (12,6))
sns.boxplot(x="month", y="cnt", data = inp2)
plt.show()


# In[36]:


inp2.groupby("month").describe()["cnt"]


# Median value for month 6 and 8 are 185 and for 7th month it is 186

# ### 7.4 Make boxplot for cnt vs. season.Which season has the highest rides in general? Expected? 

# In[37]:


plt.figure(figsize = (6,4))
sns.boxplot(x="season", y="cnt", data = inp2)
plt.show()


# In[38]:


inp2.groupby("season").describe()["cnt"]


# fall season has highest ride

# ### 7.5 Make a bar plot with the median value of cnt for each hr.Does this paint a different picture from the box plot? 

# In[39]:


sns.barplot(x="hr",y="cnt",data= inp2)
plt.show()


# ### 7.6 Make a correlation matrix for variables atemp, temp, hum, and windspeed. Which variables have the highest correlation? 

# In[40]:


variables = inp2.loc[:,["atemp","temp","hum","windspeed"]]


# In[41]:


variables.corr()


# atemp and temp are highly correlated

# ## 8. Data preprocessing
# 
# A few key considerations for the preprocessing: 
# 
# There are plenty of categorical features. Since these categorical features can’t be used in the predictive model, you need to convert to a suitable numerical representation. Instead of creating dozens of new dummy variables, try to club levels of categorical features wherever possible. For a feature with high number of categorical levels, you can club the values that are very similar in value for the target variable. 

# ### 8.1 Treating mnth column--For values 5,6,7,8,9,10, replace with a single value 5. This is because these have very similar values for cnt.Get dummies for the updated 6 mnth values
# 
# 

# In[42]:


new_data = inp2.copy()


# In[43]:


new_data.month = new_data.month.replace([6,7,8,9,10],5)


# In[44]:


new_data.month.unique()


# ### 8.2 Treating hr column--Create new mapping: 0-5: 0, 11-15: 11; other values are untouched. Again, the bucketing is done in a way that hr values with similar levels of cnt are treated the same.
# 
#  

# In[45]:


new_data.hr = new_data.hr.replace([0,1,2,3,4,5],0)
new_data.hr = new_data.hr.replace([11,12,13,14,15],11)


# In[46]:


new_data.hr.unique()


# ### 8.3 Get dummy columns for season, weathersit, weekday, mnth, and hr. You needn’t club these further as the levels seem to have different values for the median cnt, when seen from the box plots. 

# In[51]:


new_data = pd.get_dummies(new_data, columns =["season","weathersit","weekday","month","hr"], drop_first=True)


# In[55]:


new_data.columns


# ## 9. Train test split: Apply 70-30 split -- call the new dataframes df_train and df_test 

# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


df_train, df_test = train_test_split(new_data, test_size = 0.30, random_state = 40)


# ## 10. Separate X and Y for df_train and df_test. For example, you should have X_train, y_train from df_train. y_train should be the cnt column from inp3 and X_train should be all other columns.

# In[58]:


y_train = df_train.pop("cnt")
X_train = df_train


# In[59]:


y_test = df_test.pop("cnt")
X_test = df_test


# ## 11 . Model building: Use linear regression as the technique. Report the R2 on the train set 

# In[61]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train, y_train)
model


# In[64]:


from sklearn.metrics import r2_score
y_train_predict = lr.predict(X_train)
r2_score(y_train, y_train_predict)


# ## 12. Make predictions on test set and report R2. 

# In[65]:


y_test_predict = lr.predict(X_test)
r2_score(y_test, y_test_predict)


# In[ ]:




