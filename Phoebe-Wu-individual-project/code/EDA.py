
import pandas as pd
import matplotlib.pyplot as plt

bees = pd.read_csv('input/bee_data.csv')
bees.head()

#check null value
bees.isnull().sum() 

plt.figure(figsize=(8,8))
bees.date.value_counts().plot(kind = 'bar')
plt.title('Date')
plt.show()

plt.figure(figsize=(8,8))
bees.time.value_counts().plot(kind = 'bar')
plt.title('Time')
plt.show()

plt.figure(figsize=(6,6))
bees.location.value_counts().plot(kind = 'pie')
plt.title('Location')
plt.show()

bees['subspecies'].value_counts()

plt.figure(figsize=(6,6))
bees.subspecies.value_counts().plot(kind = 'pie')
plt.title('Subspecises')
plt.show()

bees['health'].value_counts()

plt.figure(figsize=(6,6))
bees.health.value_counts().plot(kind = 'pie')
plt.title('Hive Health')
plt.show()

plt.figure(figsize=(6,6))
bees.pollen_carrying.value_counts().plot(kind = 'bar')
plt.title('Pollen Carrying')
plt.show()

plt.figure(figsize=(6,6))
bees.caste.value_counts().plot(kind = 'bar')
plt.title('Caste')
plt.show()

unhealthy = bees.loc[bees['health'] != 'healthy']
#unhealthy
unhealthy.count()

healthy = bees.loc[bees['health'] == 'healthy']
healthy.count()

bees.health.value_counts()

New = bees.replace(['hive being robbed', 'few varrao, hive beetles', 'Varroa, Small Hive Beetles', 'ant problems', 'missing queen'],
             'unhealthy')

New = bees.replace(['hive being robbed', 'few varrao, hive beetles', 'Varroa, Small Hive Beetles', 'ant problems', 'missing queen'],
             'unhealthy')

plt.figure(figsize=(6,6))
New.health.value_counts().plot(kind = 'pie')
plt.title('Hive Health')
plt.show()

