


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#Get the Data



df= pd.read_csv('College_Data')
df=df.set_index(df.iloc[:,0],drop=True)
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.index.name = None
df.head()
df.info()
df.describe()


#EDA
# data visualizations!

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',height=6,aspect=1,fit_reg=False)
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',palette='coolwarm',height=6,aspect=1,fit_reg=False)
sns.set_style('darkgrid')
g=sns.FacetGrid(df,hue='Private',size=6,aspect=2)
g=g.map(plt.hist,'Outstate',alpha=0.5)

sns.set_style('darkgrid')
g=sns.FacetGrid(df,hue='Private',size=6,aspect=2)
g=g.map(plt.hist,'Grad.Rate',alpha=0.5)


df['Grad.Rate']['Cazenovia College']=100


# K Means Cluster Creation


from sklearn.cluster import KMeans

km= KMeans(n_clusters=2)
km.fit(df.drop(['Private'],axis=1))
km.cluster_centers_

df['Cluster']=df['Private'].apply(lambda x: 1 if(x=='Yes')  else 0 )


from sklearn.metrics import confusion_matrix , classification_report
print(confusion_matrix(df['Cluster'],km.labels_))
print(classification_report(df['Cluster'],km.labels_))
