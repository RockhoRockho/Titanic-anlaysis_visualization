#!/usr/bin/env python
# coding: utf-8

# # 타이타닉 데이터 분석 및 시각화
# ----

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


sns.set_style('whitegrid')


# In[3]:


titanic = sns.load_dataset('titanic')


# In[4]:


titanic.head()


# In[5]:


titanic.describe()


# In[6]:


titanic.dropna().describe()


# In[7]:


titanic.var()


# In[8]:


titanic.mad()


# In[9]:


titanic.groupby('class').count()


# In[10]:


sns.countplot(y='class', data=titanic)


# In[11]:


sns.countplot(y='sex', data=titanic)


# In[12]:


sns.countplot(y='alive', data=titanic)


# In[14]:


sns.countplot(y='alone', data=titanic)


# In[15]:


titanic.groupby('class').std()


# In[17]:


titanic.groupby('class')['fare'].median()


# In[18]:


titanic.query('alive == "yes"')


# In[19]:


titanic.query('alive == "yes"').groupby('class').count()


# In[20]:


titanic.groupby('class')['age'].describe()


# In[21]:


titanic.query("alive == 'yes'").groupby('class').describe()


# In[24]:


titanic.groupby('sex')['age'].aggregate([min, np.median, max])


# In[26]:


titanic.query("age > 30").groupby('class').median()


# In[27]:


titanic.query("fare < 20").groupby('class').median()


# In[30]:


titanic.groupby(['class', 'sex'])['age'].mean().unstack()


# In[33]:


sns.catplot(x='sex', y='age',
           hue='class', kind='bar',
           data=titanic)


# In[34]:


sns.catplot(x='who', y='age',
           hue='class', kind='bar',
           data=titanic)


# In[36]:


titanic.groupby(['class', 'sex'])['fare'].mean().unstack()


# In[37]:


titanic.groupby(['class', 'who'])['fare'].mean().unstack()


# In[38]:


sns.catplot(x='sex', y='fare',
           hue='class', kind='bar',
           data=titanic)


# In[39]:


sns.catplot(x='who', y='fare',
           hue='class', kind='bar',
           data=titanic)


# In[41]:


titanic.groupby(['class', 'sex'])['survived'].mean().unstack()


# In[42]:


titanic.pivot_table('survived', index='class', columns='sex')


# In[43]:


titanic.pivot_table('survived', index='class', columns='who')


# In[44]:


sns.catplot(x='class', y='survived',
           hue='sex', kind='bar',
           data=titanic)


# In[45]:


sns.catplot(x='class', y='survived',
           hue='who', kind='bar',
           data=titanic)


# In[46]:


age = pd.cut(titanic['age'], [0, 18, 40 ,80])
titanic.pivot_table('survived', ['sex', age], 'class')


# In[47]:


age = pd.cut(titanic['age'], [0, 18, 40 ,80])
titanic.pivot_table('survived', ['who', age], 'class')


# In[50]:


fare = pd.qcut(titanic['fare'], 3)
titanic.pivot_table('survived', ['who', age], [fare, 'class'])


# In[51]:


titanic.pivot_table('survived', index='who', columns='class', margins=True)


# In[52]:


sns.catplot(x='class', y='survived',
           col='who', kind='bar',
           data=titanic)


# In[53]:


titanic.pivot_table('survived', index='deck', columns='class', margins=True)


# In[56]:


sns.countplot(x='deck', data=titanic)


# In[57]:


sns.countplot(y='deck', hue='class', data=titanic)


# In[58]:


sns.catplot(x='survived', y='deck',
           hue='class', kind='bar',
           data=titanic)


# In[59]:


titanic.pivot_table('survived', index='embark_town', columns='class', margins=True)


# In[60]:


sns.countplot(y='embark_town', data=titanic)


# In[61]:


sns.catplot(x='survived', y='embark_town',
           hue='class', kind='bar',
           data=titanic)


# In[62]:


sns.catplot(x='sibsp', y='survived',
           kind='bar', data=titanic)


# In[63]:


sns.catplot(x='parch', y='survived',
           kind='bar', data=titanic)


# In[64]:


sns.catplot(x='alone', y='survived',
           kind='bar', data=titanic)


# In[ ]:




