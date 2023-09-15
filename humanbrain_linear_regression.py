#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('headbrain.csv')
df.head()


# In[3]:


print(df.shape)


# In[4]:


x = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values


# In[7]:


#Mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

#Total number of values
n = len(x)

#using formula to calculate to b1 and b0 or m and c
numer = 0
denom = 0
for i in range(n):
    numer += (x[i] - mean_x) * (y[i]- mean_y)
    denom += (x[i] - mean_x) ** 2

b1 = numer/ denom;
b0 = mean_y - (b1 * mean_x)

#print coefficient
print(b1, b0)


# In[10]:


# Create a scatter plot
plt.scatter(x, y, label='Data Points')

# Add the regression line to the plot
plt.plot(x, b0 + b1 * x, color='red', label='Regression Line')

# Set labels for the x and y axes
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')

# Create a legend
plt.legend()

# Show the plot
plt.show()


# In[16]:


#R square
ss_t = 0 #total sum of square
ss_r = 0 #total sum of square residuals

for i in range (n):
    y_pred = b0 + b1*x[i]
    ss_t += (y[i] - mean_y) ** 2
    ss_r += (y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)



# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = x.reshape ((n,1)) #cannot use 1-d matrics
#Creating model
reg = LinearRegression()

# Fit the model to your data
reg.fit(x, y)

#Y_prediction
y_pred = reg.predict(x)

#Calculating R square
r2_score = reg.score(x,y)
print(r2_score)


# In[ ]:




