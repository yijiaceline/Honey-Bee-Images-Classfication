
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[14]:


bees = pd.read_csv('/Users/apple/Desktop/FinalProject/input/bee_data.csv')
plt.figure(figsize=(6,6))
bees.health.value_counts().plot(kind = 'bar')
plt.title('Hive Health')
plt.show()


# In[13]:


unhealthy = bees.loc[bees['health'] != 'healthy']
unhealthy.count()


# In[12]:


healthy = bees.loc[bees['health'] == 'healthy']
healthy.count()

