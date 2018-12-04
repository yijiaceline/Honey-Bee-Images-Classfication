#check duplicates
duplicates = bees.duplicated()
duplicates[duplicates > 1]
duplicates = bees.file.value_counts()
duplicates[duplicates > 1]

plt.figure(figsize=(6,6))
bees.health.value_counts().plot(kind = 'pie')
plt.title('Hive Health')
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