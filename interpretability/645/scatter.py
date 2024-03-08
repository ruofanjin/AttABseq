import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('cnn1_split_645.csv')
plt.rcParams.update({'figure.figsize':(6.2,10), 'figure.dpi':100})
#sns.lmplot(x='true label', y='predicted label', data=df)
#f.set_xlabel('true label', fontsize=15)
#f.set_ylabel('predicted label', fontsize=15)
#sns.jointplot(x=df['true label'].tolist(),y=df['predicted label'].tolist())
#plt.legend()
#plt.savefig('645-5cv.png')
#plt.show()

sns.set(style="ticks", color_codes=True)  # darkgrid-默认, whitegrid, dark, white, ticks
x = df['true label'].tolist()
y = df['predicted label'].tolist()
g = sns.jointplot(x=x,y=y, kind="reg", space=0, truncate=False, color="#ff8884", height=7)  # xlim=(0, 10), ylim=(0, 6), kind={scatter,reg,resid,kde,hex}, shade=True, 
r, p = stats.pearsonr(x, y)
phantom, = g.ax_joint.plot([], [], linestyle="", alpha=0)
g.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])
g.savefig('split-645.png')
plt.show()