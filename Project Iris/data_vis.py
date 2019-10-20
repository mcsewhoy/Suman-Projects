import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv("iris.csv")
df.head()
df.info()


sns.catplot('name','s_length',data= df)
sns.catplot('name','s_width',data= df)
sns.catplot('name','p_length',data= df)
sns.catplot('name','p_width',data= df)

fig = plt.figure(5, figsize=(20, 20))
plt.subplot(2,2,1)
sns.scatterplot(x="s_length", y="s_width",hue="name", style="name", data=df)
plt.subplot(2,2,2)
sns.scatterplot(x="s_length", y="p_length",hue="name", style="name", data=df)
plt.subplot(2,2,3)
sns.scatterplot(x="s_width", y="p_width",hue="name", style="name", data=df)
plt.subplot(2,2,4)
sns.scatterplot(x="p_length", y="p_width",hue="name", style="name", data=df)
plt.show()