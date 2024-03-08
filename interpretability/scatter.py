import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cnn1_cv_645.csv')
plt.rcParams.update({'figure.figsize':(6.2,10), 'figure.dpi':100})
#sns.lmplot(x='true label', y='predicted label', data=df)
#f.set_xlabel('true label', fontsize=15)
#f.set_ylabel('predicted label', fontsize=15)
#sns.jointplot(x=df['true label'].tolist(),y=df['predicted label'].tolist())
#plt.legend()
#plt.savefig('645-5cv.png')
#plt.show()

import seaborn as sns
sns.set(style="darkgrid")
g = sns.jointplot(x=df['true label'].tolist(),y=df['predicted label'].tolist(), \
kind="reg", truncate=False, \
color="m", height=7)  # xlim=(0, 10), ylim=(0, 6), 
g.savefig('5cv-645.png')
plt.show()