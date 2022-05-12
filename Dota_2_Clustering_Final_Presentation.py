import pandas as pd
import seaborn as sns # plotting with seaborn
from sklearn.preprocessing import StandardScaler # scaling
from sklearn.preprocessing import MinMaxScaler # alt scaling
from sklearn.decomposition import PCA # Feature extraction
import matplotlib.pyplot as plt # plotting with pyplot
from sklearn.cluster import KMeans # calculate KMeans clustering
import sklearn.metrics as metrics # calculate silhouetted
from sklearn.manifold import TSNE # Feature dimension reduction
import time # timing TSNE
import numpy as np # for random
from scipy.spatial import distance  # calculate euclidean distance

pro_df = pd.read_csv('raw_pro_match_data.csv')
pro_df

reg_df = pd.read_csv('raw_reg_match_data.csv')
reg_df

filtered_features = ['deaths', 'kills', 'assists', 'kda', 'gold_per_min_raw',
                     'xp_per_min_raw', 'kills_per_min_raw', 'last_hits_per_min_raw']

pro_df = pro_df[filtered_features]
reg_df = reg_df[filtered_features]

# cluster centers are based on PC0, PC1, PC2, PC3

# To check if a regular player plays more like a pro
# we would calculate that player's principle components in both Pro and Reg datasets
# Then compare the euclidean distance to pro clusters and regular clusters
# shortest distance should indicate whether that player plays more like a pro than a regular player

# select a random player to test
test_player = reg_df.sample()
test_player

# remove the test player from dataframe so we can add it to the end of a new dataframe (easy to track)
reg_df = reg_df.drop(index=test_player.index.values).reset_index(drop = True)
reg_df.loc[test_player.index.values,:]

# limiting dataframe size for TSNE
subset_pro_df = pro_df.sample(frac=0.3, random_state=42).reset_index(drop = True)
subset_reg_df = reg_df.sample(frac=0.3, random_state=42).reset_index(drop = True)

# add the test player to end of both subset dataframes
subset_pro_df = pd.concat([subset_pro_df, test_player], ignore_index=True)
subset_reg_df = pd.concat([subset_reg_df, test_player], ignore_index=True)

subset_pro_df

subset_pro_df.head()

scaler = StandardScaler()

# scaling original dfs
scaled_pro_df = scaler.fit_transform(pro_df)
scaled_reg_df = scaler.fit_transform(reg_df)

# scaling subset dfs
scaled_subset_pro_df = scaler.fit_transform(subset_pro_df)
scaled_subset_reg_df = scaler.fit_transform(subset_reg_df)

# # Principle Components
# Getting Principle Component values per dataframe
def create_pc_df(data_frame):
    pca = PCA()
    pc = pca.fit_transform(data_frame)
    pc_cols = []
    for i in range(len(pro_df.columns)):
        pc_cols.append(f'PC{i}')

    # printing the cumulative sum of variance of each Principle Component
    print(pca.explained_variance_ratio_.cumsum())
    
    # making a plot of the cumulative sum of variance
    plt.figure(figsize = (10, 8))
    plt.plot(range(1, 9), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    
    return(pd.DataFrame(data=pc, columns=pc_cols))
    
# calling function
pro_pc_df = create_pc_df(scaled_pro_df)
subset_pro_pc_df = create_pc_df(scaled_subset_pro_df)

reg_pc_df = create_pc_df(scaled_reg_df)
subset_reg_pc_df = create_pc_df(scaled_subset_reg_df)

print(subset_pro_pc_df.describe())
print(subset_reg_pc_df.describe())

# 4 principle components explain 94% variance
# set pc dataframe to first 4 columns
pro_pc_df = pro_pc_df.iloc[:, :4]
reg_pc_df = reg_pc_df.iloc[:, :4]
sub_pro_pc_df = subset_pro_pc_df.iloc[:, :4]
sub_reg_pc_df = subset_reg_pc_df.iloc[:, :4]

sub_reg_pc_df

# adding the Principle components to the original dataframes
# reg_df = pd.concat([reg_df, reg_pc_df], axis=1)
# pro_df = pd.concat([pro_df, pro_pc_df], axis=1)
subset_reg_df = pd.concat([subset_reg_df, sub_reg_pc_df], axis=1)
subset_pro_df = pd.concat([subset_pro_df, sub_pro_pc_df], axis=1)

subset_reg_df

# T-SNE
# feature dimension reduction makes plotting visually better
def get_tsne(df, n):
    return TSNE(n_components=n,random_state=42).fit_transform(df)

startTime = time.time()

# takes a long time so for now we are testing with a random sample
# getting 3 TSNE components for 3d plot
sub_pro_tsne = get_tsne(sub_pro_pc_df, 3)
sub_reg_tsne = get_tsne(sub_reg_pc_df, 3)

executionTime = (time.time() - startTime)

print('Execution time in seconds: ' + str(executionTime))

# converting np array to dataframe
pro_tsne_df = pd.DataFrame(sub_pro_tsne, columns = ['TSNE0', 'TSNE1', 'TSNE2'])
reg_tsne_df = pd.DataFrame(sub_reg_tsne, columns = ['TSNE0', 'TSNE1', 'TSNE2'])
pro_tsne_df

# concating dataframes
subset_reg_df = pd.concat([subset_reg_df, reg_tsne_df], axis=1)
subset_pro_df = pd.concat([subset_pro_df, pro_tsne_df], axis=1)

subset_pro_df

# # KMeans: Calculating k clusters using elbow method and silhouette
K = range(1, 12)
wss = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++', random_state = 42)
    km = km.fit(sub_reg_pc_df)
    # within-cluster SSE given by km.inertia_
    wss_score = km.inertia_
    wss.append(wss_score)
    
# Elbow Method for reg pca

cluster_centers = pd.DataFrame({'Clusters' : K, 'WSS' : wss})

sns.lineplot(x = 'Clusters', y = 'WSS', data = cluster_centers, marker = 'o')

# Sihoulette method for reg pca

for i in range(3, 10):
    labels = KMeans(n_clusters = i, init = 'k-means++', random_state = 200).fit(sub_reg_pc_df).labels_
    print("Silhouette score for k(clusters = "+str(i)+" is " 
    +str(metrics.silhouette_score(sub_reg_pc_df, labels, metric = 'euclidean', sample_size = 1000, random_state = 200)))

K = range(1, 12)
wss = []
for k in K:
    km = KMeans(n_clusters=k, init='k-means++', random_state = 42)
    km = km.fit(sub_pro_pc_df)
    # within-cluster SSE given by km.inertia_
    wss_score = km.inertia_
    wss.append(wss_score)

# Elbow Method for pro pca

cluster_centers = pd.DataFrame({'Clusters' : K, 'WSS' : wss})

sns.lineplot(x = 'Clusters', y = 'WSS', data = cluster_centers, marker = 'o')

# Sihoulette method for pro pca

for i in range(3, 10):
    labels = KMeans(n_clusters = i, init = 'k-means++', random_state = 200).fit(sub_pro_pc_df).labels_
    print("Silhouette score for k(clusters = "+str(i)+" is " 
    +str(metrics.silhouette_score(sub_pro_pc_df, labels, metric = 'euclidean', sample_size = 1000, random_state = 200)))

# Silhouette suggests three or four clusters, elbow method suggests three.
# We will use three clusters.

km_pro = KMeans(n_clusters = 3, init='k-means++', random_state = 42)

km_pro.fit(sub_pro_pc_df)

km_reg = KMeans(n_clusters = 3, init='k-means++', random_state = 42)

km_reg.fit(sub_reg_pc_df)

# adding Cluster classifcation to dataframes
subset_pro_df['Cluster'] = km_pro.labels_
subset_reg_df['Cluster'] = km_reg.labels_

subset_pro_df

# 3D PLOT
# plotting random points
np.random.seed(42)
rndperm_pro = np.random.permutation(subset_pro_df.shape[0])
rndperm_reg = np.random.permutation(subset_reg_df.shape[0])

# Comparing 3D plots of PCA VS. TSNE
# plot 3d of pro PCA
ax = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')

ax.set_title('3D plot of Pro Players (PCA)')
propca_scatter = ax.scatter(
    xs=subset_pro_df.loc[:]['PC0'], 
    ys=subset_pro_df.loc[:]['PC1'], 
    zs=subset_pro_df.loc[:]['PC2'], 
    c=subset_pro_df.loc[:]['Cluster'], 
    cmap='tab10', 
)
ax.set_xlabel('PCA 0')
ax.set_ylabel('PCA 1')
ax.set_zlabel('PCA 2')

legend = ax.legend(*propca_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.show()

# plot 3d of regular PCA
ax = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')

ax.set_title('3D plot of Regular Players (PCA)')

regpca_scatter = ax.scatter(
xs=subset_reg_df.loc[:]['PC0'], 
ys=subset_reg_df.loc[:]['PC1'], 
zs=subset_reg_df.loc[:]['PC2'], 
c=subset_reg_df.loc[:]['Cluster'], 
cmap='tab10'
)
ax.set_xlabel('PCA 0')
ax.set_ylabel('PCA 1')
ax.set_zlabel('PCA 2')

legend = ax.legend(*regpca_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.show()

# plot 3d of pro TSNE
ax = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')

ax.set_title('3D plot of Pro Players (TSNE)')
protsne_scatter = ax.scatter(
    xs=subset_pro_df.loc[rndperm_pro,:]['TSNE0'], 
    ys=subset_pro_df.loc[rndperm_pro,:]['TSNE1'], 
    zs=subset_pro_df.loc[rndperm_pro,:]['TSNE2'], 
    c=subset_pro_df.loc[rndperm_pro, :]['Cluster'], 
    cmap='tab10'
)
ax.set_xlabel('TSNE 0')
ax.set_ylabel('TSNE 1')
ax.set_zlabel('TSNE 2')

legend = ax.legend(*protsne_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.show()

# plot 3d of reg TSNE
ax = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.set_title('3D plot of Regular Players (TSNE)')
proregtsne_scatter = ax.scatter(
    xs=subset_reg_df.loc[rndperm_reg,:]['TSNE0'], 
    ys=subset_reg_df.loc[rndperm_reg,:]['TSNE1'], 
    zs=subset_reg_df.loc[rndperm_reg,:]['TSNE2'], 
    c=subset_reg_df.loc[rndperm_reg, :]['Cluster'], 
    cmap='tab10'
)
ax.set_xlabel('TSNE 0')
ax.set_ylabel('TSNE 1')
ax.set_zlabel('TSNE 2')

legend = ax.legend(*proregtsne_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.show()

# 2D PLOT 
# Plotting 2d
startTime = time.time()

# getting 2 TSNE components for 2d plot
d2_sub_pro_tsne = get_tsne(sub_pro_pc_df, 2)
d2_sub_reg_tsne = get_tsne(sub_reg_pc_df, 2)

executionTime = (time.time() - startTime)

print('Execution time in seconds: ' + str(executionTime))

d2_pro_tsne_df = pd.DataFrame(d2_sub_pro_tsne, columns = ['2D-TSNE0', '2D-TSNE1'])
d2_reg_tsne_df = pd.DataFrame(d2_sub_reg_tsne, columns = ['2D-TSNE0', '2D-TSNE1'])

subset_reg_df = pd.concat([subset_reg_df, d2_reg_tsne_df], axis=1)
subset_pro_df = pd.concat([subset_pro_df, d2_pro_tsne_df], axis=1)

subset_pro_df

# Comparing 2D PCA VS 2D TSNE
fig = plt.figure(figsize=(14,14))

reg2dpca_scatter = plt.scatter(subset_pro_df[:]['PC0'], subset_pro_df[:]['PC1'], c=subset_pro_df['Cluster'], alpha=0.5)
plt.xlabel('PC0')
plt.ylabel('PC1')

legend = plt.legend(*reg2dpca_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.title('2D PCA Plot of Pro clusters (PCA)', fontsize=20);

fig = plt.figure(figsize=(14,14))

pro2dtsne_scatter = plt.scatter(subset_pro_df[:]['2D-TSNE0'], subset_pro_df[:]['2D-TSNE1'], c=subset_pro_df['Cluster'], alpha=0.5)
plt.xlabel('2D-TSNE0')
plt.ylabel('2D-TSNE1')

legend = plt.legend(*pro2dtsne_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.title('2D PLOT of Pro clusters (TSNE)', fontsize=20);

fig = plt.figure(figsize=(14,14))

reg2dpca_scatter = plt.scatter(subset_reg_df[:]['PC0'], subset_reg_df[:]['PC1'], c=subset_reg_df['Cluster'], alpha=0.5)
plt.xlabel('PC0')
plt.ylabel('PC1')

legend = plt.legend(*reg2dpca_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.title('2D PCA Plot of Regular clusters (PCA)', fontsize=20);

fig = plt.figure(figsize=(14,14))

reg2dtsne_scatter = plt.scatter(subset_reg_df[:]['2D-TSNE0'], subset_reg_df[:]['2D-TSNE1'], c=subset_reg_df['Cluster'], alpha=0.5)
plt.xlabel('2D-TSNE0')
plt.ylabel('2D-TSNE1')

legend = plt.legend(*reg2dtsne_scatter.legend_elements(), loc="upper right", title='Clusters')

plt.title('2D PLOT of Regular clusters (TSNE)', fontsize=20);

km_pro.cluster_centers_

km_reg.cluster_centers_

# cluster centers are based on PC0, PC1, PC2, PC3

# To check if test_player plays more like a pro
# we would calculate that player's principle components in both Pro and Reg datasets
# Then compare the euclidean distance of test_player to both clusters in pro and reg dataframes
# shortest distance should indicate whether that player plays more like a pro than a regular player

# get the updated information on test_player per dataframe
test_player_pro = subset_pro_df.tail(1)
test_player_reg = subset_reg_df.tail(1)

test_pro_point = test_player_pro[['PC0', 'PC1', 'PC2', 'PC3']].to_numpy()
test_reg_point = test_player_reg[['PC0', 'PC1', 'PC2', 'PC3']].to_numpy()
print(test_reg_point)
print(test_pro_point)

# get coordinates of pro cluster center that the test player is in
pro_centroid = km_pro.cluster_centers_[test_player_pro['Cluster'].values[0]]

# coordinates of reg cluster the test player is in
reg_centroid = km_reg.cluster_centers_[test_player_reg['Cluster'].values[0]]

print(reg_centroid)
print(pro_centroid)

# Calculating distance to centroids
pro_distance = distance.euclidean(test_pro_point, pro_centroid)
reg_distance = distance.euclidean(test_reg_point, reg_centroid)

print('pro distance:', pro_distance)
print('reg distance:', reg_distance)

if (pro_distance < reg_distance):
    print('The test player plays more like the average pro player')
else:
    print('The test player plays more like the average regular (high skill) player')

# plot histograms of each feature per cluster

def get_histograms(df, indexes = []):
    for cluster in df:
        grid = sns.FacetGrid(df, col='Cluster')
        grid.map(plt.hist, cluster)
        
get_histograms(subset_pro_df[['deaths', 'kills', 'assists', 'kda', 'gold_per_min_raw', 'xp_per_min_raw', 'kills_per_min_raw', 'last_hits_per_min_raw', 'Cluster']])

# # From this I can infer that clusters for Pro players:
# 
# - cluster 2 = Core players (Position 1-3)
# - cluster 0 = Support players (Position 4-5)
# - cluster 1 = Inconclusive
# - cluster 1 could include high performing supports or underperforming cores or Position 3

get_histograms(subset_reg_df[['deaths', 'kills', 'assists', 'kda', 'gold_per_min_raw', 'xp_per_min_raw', 'kills_per_min_raw', 'last_hits_per_min_raw', 'Cluster']])

# # From this I can infer that clusters for Regular players:
# 
# - cluster 1 = Core players (Position 1-3)
# - cluster 0 = Support players (Position 4-5)
# - cluster 2 = Inconclusive
# - cluster 2 could include high performing supports or underperforming cores or Position 3

print(test_player_pro['Cluster'])
print(test_player_reg['Cluster'])

# From this we can infer that test_player plays the core role. More like the average pro player than an average high skill player