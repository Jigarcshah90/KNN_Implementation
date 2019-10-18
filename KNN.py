#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Simple implementation of the KNN using IRIS Dataset.

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
import seaborn as sns


# In[2]:


dataset = pd.read_csv('IRIS.csv')


# In[3]:


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values


# In[4]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[5]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


# In[6]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=1)  
classifier.fit(X_train, y_train)


# In[7]:


y_pred = classifier.predict(X_test)


# In[8]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


# In[9]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[10]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  


# In[11]:


df = pd.DataFrame()
df['Pred_plot'] = [5.9,3.1,0.4,2.1]
Pred_test = scaler.transform([[5.9,3.1,0.4,2.1]])
pred = classifier.predict(Pred_test) 
pred


# In[15]:


knnn = KNeighborsClassifier(n_neighbors=2)
knnn.fit(X_train, y_train)

## Testing the Model
df = pd.DataFrame()
df['Pred_plot'] = [6.0,3.1,5.4,2.1]
Pred_test = scaler.transform([[6.0,3.1,5.4,2.1]])
pred = classifier.predict(Pred_test) 
pred


# In[ ]:




