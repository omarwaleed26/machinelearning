import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot  as plt 
import pandas as  pd 
import  numpy as np
import sklearn.preprocessing as prep
from  sklearn.datasets import  make_blobs 
from sklearn.neighbors  import NearestNeighbors 
from sklearn.cluster  import DBSCAN
from  sklearn.cluster  import  KMeans 
from  sklearn.mixture  import GaussianMixture
from  sklearn.metrics  import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
from sklearn import datasets
Data=pd.read_csv('Customer Data.csv')
[rows,columns]=np.shape(Data)
data=Data.iloc[:,1:columns]
New_data=data[["Age", "Income"]]
X=pd.DataFrame(New_data).to_numpy()
#kmeans
def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk' 
    alpha = 0.5 
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)
display_cluster(X)
#kmean
plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
n_bins = 6
centers = [(-3, -3), (0, 0), (5,2.5),(-1, 4), (4, 6), (9,7)]
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 15)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_
for key, val in mapping1.items():
    print(f'{key} : {val}')
plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('K values')
plt.ylabel('Distortion')
plt.title('The Elbow Method')
plt.show()
kmeans = KMeans(n_clusters=8).fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14]
scores=[]
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    scores=np.append(scores,silhouette_avg)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7,)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10 
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
K_used=np.argmax(scores)+2
Best_score=np.max(scores)
plt.show()
#dbscan
dbscan_score=[];
score=0;
k_clusters=[];
all_scores=[];
X = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.5,min_samples=5) 
db.fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) 
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


epsilon=np.arange(0.1,3,0.1)
min_samples=np.arange(5,25,1)

dbscan_score=[];
score=0;
k_clusters=[];
all_scores=[];





epsilon=np.arange(0.1,3,0.1)
min_samples=np.arange(5,25,1)

dbscan_score=[];
score=0;
k_clusters=[];
all_scores=[];
for s in min_samples:
    dbscan_score=[];
    for e in epsilon:
        db = DBSCAN(eps=e, min_samples=s).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        

        n_clusters_ = len(set(labels)) 
        

        print("Estimated number of clusters: %d" % n_clusters_)
        if  n_clusters_!=1:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
            score=metrics.silhouette_score(X, labels);
        else:
            score=0
        dbscan_score.append(score)
        all_scores.append(score)
        k_clusters.append(n_clusters_)
    plt.figure()
    plt.plot(epsilon.flatten(),dbscan_score)
   
index=np.argmax(all_scores)
max_score=np.max(all_scores)

best_score_dbscan=all_scores[index]

dbscan_best_k=k_clusters[index]


plt.figure()

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
      
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % dbscan_best_k)
plt.show()

#hierarical
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
  

X_normalized = normalize(X_scaled)
  

X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

plt.figure(figsize =(8, 8))
plt.title('Visualising the data dendogram')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))
distance_th=np.arange(0.1,2,1)
silhouette_scores = []
for i in range(np.size(distance_th)):
    Hi1=AgglomerativeClustering(n_clusters =None,affinity='euclidean',linkage='single',distance_threshold=distance_th[i])
    Hi1_p=Hi1.fit_predict(X_principal)
    Hi1_clusters=np.max(Hi1_p)+1
    if Hi1_clusters!=1:
        silhouette_scores.append(silhouette_score(X_principal, Hi1.fit_predict(X_principal)))
        plt.figure(figsize =(6, 6))
        plt.title('euclidean')
        plt.scatter(X_principal['P1'], X_principal['P2'], c = Hi1.fit_predict(X_principal), cmap ='rainbow')
        plt.show()

   
    Hi2=AgglomerativeClustering(n_clusters =None,affinity='cosine',linkage='average',distance_threshold=distance_th[i])
    Hi2_p=Hi2.fit_predict(X_principal)
    Hi2_clusters=np.max(Hi2_p)+1
    if Hi2_clusters!=1:
        silhouette_scores.append(silhouette_score(X_principal, Hi2.fit_predict(X_principal)))
        plt.figure(figsize =(6, 6))
        plt.title('cosine')
        plt.scatter(X_principal['P1'], X_principal['P2'], c = Hi2.fit_predict(X_principal), cmap ='rainbow')
        plt.show()
    
    Hi3=AgglomerativeClustering(n_clusters = None,affinity='manhattan',linkage='average',distance_threshold=distance_th[i])
    Hi3_p=Hi3.fit_predict(X_principal)
    Hi3_clusters=np.max(Hi3_p)+1
    if Hi3_clusters!=1:
        silhouette_scores.append(silhouette_score(X_principal, Hi3.fit_predict(X_principal)))
        plt.figure(figsize =(6, 6))
        plt.title('manhattan')
        plt.scatter(X_principal['P1'], X_principal['P2'], c = Hi3.fit_predict(X_principal), cmap ='rainbow')
        plt.show()
final_score_=np.max(silhouette_scores)
#gaussian
d = pd.DataFrame(X)
 

plt.scatter(d[0], d[1])
gmm = GaussianMixture(n_components = 3,covariance_type='tied')
gmm.fit(d)
 

labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 

plt.figure()
plt.title('tied')
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')


gmm = GaussianMixture(n_components = 3,covariance_type='full')
gmm.fit(d)
 

labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 

plt.figure()
plt.title('full')
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')



gmm = GaussianMixture(n_components = 3,covariance_type='spherical')
gmm.fit(d)
 

labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 

plt.figure()
plt.title('spherical')
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')


gmm = GaussianMixture(n_components = 3,covariance_type='diag')
gmm.fit(d)
 

labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 
plt.figure()
plt.title('diag')
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')


x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T

GMM = GaussianMixture(n_components=3).fit(X) 
print('Converged:',GMM.converged_) 
means = GMM.means_ 
covariances = GMM.covariances_



Y = np.array([[0.5],[0.5]])
prediction = GMM.predict_proba(Y.T)
print(prediction)

   
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(X[:,0],X[:,1])
ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
for m,c in zip(means,covariances):
    multi_normal = multivariate_normal(mean=m,cov=c)
    ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
plt.title('2d contour')    
plt.show()
print('kmeans best:',Best_score)
print('dbscan best:',max_score)
print('Hierarical best',final_score_)
print('best d:',distance_th[i])
