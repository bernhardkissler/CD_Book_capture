

# %%
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import plotly.express as px
from scipy.cluster.vqscipy.cluster.vq import vq, kmeans, whiten
from sklearn.neighbors.kde import KernelDensity


#%%
df = pd.read_csv("cds.csv")
df["avg_y"] = (df["upper_left_y"] + df["upper_right_y"] + df["bottom_right_y"] + df["bottom_left_y"])/4
df["avg_x"] =  0
# (df["upper_left_x"] + df["upper_right_x"] + df["bottom_right_x"] + df["bottom_left_x"])/4
df.head()

#%%
fig = px.scatter(df, y="avg_x", x="avg_y", color="word")
fig.show()

# %%
features = df[["avg_x", "avg_y"]]
centroids, x = kmeans(features,11)
centroids[0]
df_centroids = pd.DataFrame(centroids, columns=["cent_x", "cent_y"])
fig = px.scatter(df_centroids, x="cent_x", y="cent_y")
fig.show()

#%%

avg_y_sort = df["avg_y"].sort_values(ascending=True)
fig = px.histogram(avg_y_sort, x=avg_y_sort, nbins=100)
fig.show()

#%%
from sklearn.neighbors.kde import KernelDensity