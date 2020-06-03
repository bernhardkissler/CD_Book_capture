# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.vq import vq, kmeans
from scipy.signal import argrelextrema
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans


#%%
# Read in Data and calculate averages
df = pd.read_csv("cds.csv")
df["avg_y"] = (
    df["upper_left_y"]
    + df["upper_right_y"]
    + df["bottom_right_y"]
    + df["bottom_left_y"]
) / 4
df["avg_x"] = (
    df["upper_left_x"]
    + df["upper_right_x"]
    + df["bottom_right_x"]
    + df["bottom_left_x"]
) / 4
df["x_0"] = 0
df.head()

# isolate data for clustering
features = df[["x_0", "avg_y"]]

#%%
# Scatter plot of initial distr
fig = px.scatter(df, y="avg_x", x="avg_y", color="word")
fig.show()

#%%
# Histogramm of initial distr
avg_y_sort = df["avg_y"].sort_values(ascending=True)
fig = px.histogram(avg_y_sort, x=avg_y_sort, nbins=100)
fig.show()

# %%
# SCIPY k means estimation
centroids, x = kmeans(features, 11)
centroids[0]
df_centroids = pd.DataFrame(centroids, columns=["cent_x", "cent_y"])
fig = px.scatter(df_centroids, y="cent_x", x="cent_y")
fig.show()

#%%
# SCIKIT k means estimation
sk_kmeans = KMeans(n_clusters=11, random_state=0).fit(features)
len(sk_kmeans.labels_)
skcentroids = sk_kmeans.cluster_centers_
df_kmeans = pd.DataFrame(skcentroids, columns=["cent_x", "cent_y"])
fig = px.scatter(df_kmeans, y="cent_x", x="cent_y")
fig.show()

#%%
# Calculate kernel density estimation (gaussian or exponential) and its minima and maxima
a = np.array(avg_y_sort).reshape(-1, 1)
kde = KernelDensity(kernel="exponential", bandwidth=3).fit(a)
s = np.linspace(0, 3000)
e = kde.score_samples(s.reshape(-1, 1))
mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

#%%
# plotting kde fit with minima and maxima
fig = px.line(x=s, y=e,)
fig.add_trace(go.Scatter(x=s[ma], y=e[ma], mode="markers", name="maxima"))
fig.add_trace(go.Scatter(x=s[mi], y=e[mi], mode="markers", name="minima"))
fig.show()
