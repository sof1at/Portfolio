# # Importing Libararies
# import matplotlib.pyplot as plt
# import seaborn as sns


# years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
# apples = [0.89, 0.72, 0.82, 0.49, 0,3, 0.79]
# oranges = [0.5, 0.6, 0.34, 0.89, 0.63, 0.55, 0.98]
# plt.plot(years, apples)
# plt.plot(years, oranges)
# plt.xlabel('Years')
# plt.ylabel('Apples')
# plt.title('Apples Production Over Years')
# plt.legend(['apples'])
# plt.show()
#plt.tite('Iris Datset Scatter Plot')

#Showing the plot
#plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# tips_df = sns.load_dataset("tips")
# print(tips_df)
# sns.barplot(x="day", y="total_bill",hue ='sex', data=tips_df)
# plt.show()

#Showing the plot
#plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# flo_df = sns.load_dataset('iris')

# plt.title('Iris Dataset histogram')
# plt.hist(flo_df.petal_length, bins=5)
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

flo_df = sns.load_dataset('iris')

 

# sns.pairplot(flo_df, hue='species')
# plt.show()

# Calculate the correlation matrix
# Convert the 'species' column to numerical values
flo_df['species'] = pd.factorize(flo_df['species'])[0]
correlation_matrix = flo_df.corr()
# Create a heatmap
sns.heatmap(correlation_matrix, annot=True)
# Display the heatmap
plt.show()