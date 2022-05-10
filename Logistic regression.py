#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[26]:


bank=pd.read_csv("C:\\Users\\Shivani\\OneDrive\\Desktop\\bank.csv")
bank


# In[27]:


bank.info()


# In[28]:


data1=pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome','month'])
data1


# In[30]:


pd.set_option("display.max.columns", None)
data1


# In[31]:


data1.info()


# In[32]:


data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
data1['y'] = np.where(data1[''].str.contains("yes"), 1, 0)
data1


# In[33]:


x=pd.concat([data1.iloc[:,0:10],data1.iloc[:,11:]],axis=1)
y=data1.iloc[:,10]


# In[34]:


classifier=LogisticRegression()
classifier.fit(x,y)


# In[35]:


y_pred=classifier.predict(x)
y_pred


# In[17]:


y_pred_df=pd.DataFrame({'actual_y':y,'y_pred_prob':y_pred})
y_pred_df


# In[19]:


(39155+1167)/(39155+767+4122+1167)


# In[20]:


classifier.predict_proba(x)[:,1]


# In[38]:


fpr,tpr,thresholds=roc_curve(y,classifier.predict_proba(x)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)

