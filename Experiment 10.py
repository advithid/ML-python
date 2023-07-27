#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()


# In[2]:


dir(iris)


# In[3]:


iris.feature_names


# In[4]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[5]:


df['target']=iris.target
df.head()


# In[6]:


iris.target_names


# In[7]:


df[df.target==1].head()


# In[8]:


df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df.head()


# In[9]:


from matplotlib import pyplot as plt


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]


# In[12]:


df0.head()


# In[13]:


df1.head()


# In[14]:


df2.head()


# In[15]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='.')


# In[16]:


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.')


# In[17]:


df1.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x=df.drop(['target','flower_name'],axis='columns')


# In[20]:


x.head()


# In[21]:


y=df.target
y


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[23]:


len(x_train)


# In[24]:


len(x_test)


# In[25]:


from sklearn.svm import SVC 
model=SVC()


# In[26]:


model.fit(x_train,y_train)


# In[27]:


model.score(x_test,y_test)


# In[ ]:




