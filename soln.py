#!/usr/bin/env python
# coding: utf-8

# In[43]:


#Create own Dataframe using pandas' constructor. 
#Dataframe have column has row_no, team, player_name, run

import random
import string
import pandas as pd

data = {
    'team':[],
    'player_name':[],
    'run': [],
}
total_player = 1500
max_match = 500
for i in range(total_player):
    data['team'].append(random.randint(1,10))
    data['player_name'].append(''.join(random.choices(string.ascii_uppercase, k=6)))
    t_match = random.randint(0,max_match)
    data['run'].append(random.randint(0,250))
        
df = pd.DataFrame(data)
df


# In[44]:


#print records of players who scored more than 50 runs. 
d = df[(df['run']>50)]
d


# In[45]:


#sorting the records based on the number of runs scored by each player
d1=df
d1.sort_values(by=['run'])


# In[46]:


#calculating the percentage runs scored by each player, where the percentage is with respect to the total runs scored by the team in an innings. 


  
# Calculate Percentage 
df['percent'] = (df['run'] / 
                      df['run'].sum()) * 100
# Show the dataframe 
df


# In[47]:


#Add a new column to the Dataframe. For example, you may want to add the number written on the player's jersey. 
jersey_no = {}
for i in range(total_player):
    jersey_no[df['player_name'][i]] = random.randint(1,100)
    
df['jersey_no'] = df['player_name'].map(jersey_no)
print('\nAdding new colum jersey to each player.')
df


# In[48]:


#Export the Dataframe to a file in pickle format and save it as a file.
df_p= df
df_p.to_pickle("./dummy.pkl")


# In[49]:


#Read the pickle file back to the program
df_up = pd.read_pickle("./dummy.pkl")
df_up


# In[ ]:





# In[50]:


'''Use matplotlib library to make a scatter plot of columns that contain numeric
data. Provide labels to the axes. '''

import matplotlib.pyplot as plt
import numpy as np

X= df['jersey_no'].values
Y = df['run'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

plt.scatter(X,Y,label='scatter')
plt.xlabel('jersey_no')
plt.ylabel('run')
plt.legend()
plt.show()


# In[ ]:





# In[51]:


''' Implement linear regression to model the dependency between two variables - the predictor total run and target run'''
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i]- mean_x) **2

w1  = numer/denom
w0  = mean_y - (w1*mean_x)

# print coefficient
print('Coefficients are: ',w0,w1)


# In[ ]:





# In[52]:


# Plot the linear regression line
max_x = np.max(X)
min_x = np.min(X)

x = np.linspace(min_x,max_x)
y = w0 + w1*x

# ploting scatter line
plt.scatter(X,Y, label='scatter')

# ploting regression line
plt.plot(x,y,color='red', label='Regression line')


plt.xlabel('jersey_no')
plt.ylabel('Run')
plt.legend()
plt.show()


# In[53]:


# R^2 method to check correctness

ss_t = 0
ss_r = 0
for i in range(n):
    y_pred = w0 + w1 * X[i]
    ss_t += (Y[i] - mean_y) **2
    ss_r += (Y[i] - y_pred) **2
r2 = 1 - (ss_r/ss_t)
print('scores are ',r2)


# In[ ]:





# In[ ]:





# In[54]:


# LINEAR REGRESSION USING THE SKLEARN
print("Comparing result with sklearn fit()")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((n,1))

# create model
reg = LinearRegression()

# fitting training data
reg = reg.fit(X,Y)

# Y prediction
y_pred = reg.predict(X)

# cal R2 error
r2_score = reg.score(X,Y)

print('coefficients are: ', reg.coef_)
print('scores are ',r2_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




