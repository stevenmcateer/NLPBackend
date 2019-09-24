import pymc3 as pm
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

df = load_iris(return_X_y=False)
df = pd.DataFrame(data=np.c_[df['data'], df['target']],
                     columns=df['feature_names'] + ['target'])

df = df.rename(index=str, columns={'sepal length (cm)': 'sepal_length', 'sepal width (cm)': 'sepal_width', 'target': 'species'})

df = df.loc[df['species'].isin([0, 1])]
y = pd.Categorical(df['species']).codes
x = df[['sepal_length', 'sepal_width']].values


# working ordered logit
x = df['sepal_length'].values
y = pd.Categorical(df['species']).codes

with pm.Model() as model:
    cutpoints = pm.Normal("cutpoints", mu=[-2,2], sd=10, shape=2,
                          transform=pm.distributions.transforms.ordered)

    y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=x, observed=y)
    tr = pm.sample(1000)