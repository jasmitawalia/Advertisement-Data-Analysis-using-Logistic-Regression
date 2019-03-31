
# coding: utf-8

# ##Advertisement dataset project
# 
# In this project we will be working with a advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Importing Libraries
# 

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Getting the Data
# 

# In[3]:

ad_data = pd.read_csv('advertising.csv')


# #

# In[4]:

ad_data.head()


# ** Using info and describe() on out ad_data**

# In[5]:

ad_data.info()


# In[6]:

ad_data.describe()


# ## analysing our data using seaborn now
# 
# 
# 

# In[7]:

sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# **this is a jointplot showing Area Income versus Age.**

# In[8]:

sns.jointplot(x='Age',y='Area Income',data=ad_data)


# ** Daily Time spent on site vs. Age.**

# In[9]:

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='blue',kind='kde');


# **'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[10]:

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')


# #

# In[11]:

sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# ## Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 

# ** Spliting the data into training set and testing set using train_test_split**

# In[12]:

from sklearn.model_selection import train_test_split


# In[13]:

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[14]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# #

# In[15]:

from sklearn.linear_model import LogisticRegression


# In[16]:

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now we will predict values for the testing data.**

# In[20]:

predictions = logmodel.predict(X_test)


# ** Creating a classification report for the model.**

# In[21]:

from sklearn.metrics import classification_report


# In[22]:

print(classification_report(y_test,predictions))


# ## thank you.

# In[23]:

predictions


# In[24]:

y_test


# In[ ]:



