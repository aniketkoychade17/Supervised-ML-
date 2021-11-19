#!/usr/bin/env python
# coding: utf-8

# # Predicting the Percentage of a Student based on Number of study hours

# In[1]:


# importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading and importing data 
data=pd.read_csv('http://bit.ly/w-data')
data.head(15)


# To see the co-relation between 'hours' and 'scores' we have to plot 2-D graph between them. To show the distribution we have to use matplotlib library.

# In[3]:


# plotting the distribution 
data.plot(x='Hours',y='Scores',style='s',grid='ON')
plt.title('Hours Vs Percentage')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()


# **So, From the above plot we can clearly see there is a positive linear relation berween Number of hours studied and gained percentages.**
# 

# Now, we have to splits the data into input and output sets. As we are predicting the Percentage gained on the basis of Number of hours student studied, So In this case, our input set must be hours coloumn and output set must be Scores.

# In[4]:


# Input dataset stored in X
#X = data['Hours']

# Output dataset stored in y
#y = data['Scores']
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# By now, We have split our data into input and output dataset. Now its time to further split our data into training and test sets. To do this we are using Scikit-Learn's built-in train_test_split() method.

# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# Here, We are allocating 20% data to test and remaining 80% data to train our model.

# ### Algorithm Training

# In[6]:


from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train)


# Hence, We have successfully completed our Algorithm Training.

# In[7]:


# Plotting the regression line
line = model.coef_*X+model.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ## Predictions
# 
# Now, it's time to make predictions using our trained data.

# In[17]:


print(X_test)    #This is our testing data in hours
y_pre = model.predict(X_test)    #Based on number of studied hours predicting the Scores


# In[18]:


new_data=pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pre})
new_data


# In[24]:


hours = 9.5
mod=np.array([hours])
mod=mod.reshape(-1,1)
own_pre = model.predict(mod)
print('No of Hours = {}'.format(hours))
print('Predicted Score = {}'.format(own_pre[0]))


# In[15]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pre)) 


# In[ ]:




