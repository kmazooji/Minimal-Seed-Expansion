import numpy as np
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import datasets
import warnings
import time
from sklearn.datasets import fetch_openml
warnings.filterwarnings("ignore")
from MSE import MSE


def plot_MSE(X, labels, Npts, K, M, D, A_init, A_step, A_max_iter, expansion_Npts, disp_2feat, disp_TSNE):
    
    s = MSE(     Npts=Npts,
                 K = K,
                 M = M,
                 D = D,
                 A_init = A_init,
                 A_step = A_step, 
                 A_max_iter = A_max_iter,
                 expansion_Npts = expansion_Npts
                 )
    
    t1 = time.time()
    clustering = s.fit_predict(X)
    t2 = time.time()
    
    print("")

    y_pred = clustering 

    extracted_clusters = {}
    for i in range(y_pred.shape[0]):
        if y_pred[i] not in extracted_clusters:
            extracted_clusters[y_pred[i]] = [X[i, :]]
        else:
            extracted_clusters[y_pred[i]].append(X[i, :])


    for c in extracted_clusters:
        if c >= 0: 
            print("cluster " + str(int(c)) + ": " + str(len(extracted_clusters[c])))
    
    ARI = metrics.adjusted_rand_score(labels, y_pred)
    NMI = metrics.normalized_mutual_info_score(labels, y_pred)
    print("ARI:", ARI)
    print("NMI:", NMI)
    print("time taken:", t2 - t1)


    plot1 = True
    if plot1:
        if disp_2feat:
            for c in extracted_clusters: 
                if c == -1:
                    plt.scatter([p[0] for p in extracted_clusters[c]], [p[1] for p in extracted_clusters[c]], label = 'noise')
                else:  
                    plt.scatter([p[0] for p in extracted_clusters[c]], [p[1] for p in extracted_clusters[c]])
            
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.legend(loc = 'lower right')
            plt.title("Spectral (first two features)")
            plt.show()  

        if disp_TSNE:
            X_tsne = TSNE(n_components=2).fit_transform(X)
            for c in extracted_clusters:    
                if c == -1:
                    plt.scatter([X_tsne[point_to_index[tuple(p)], 0] for p in extracted_clusters[c]], [X_tsne[point_to_index[tuple(p)], 1] for p in extracted_clusters[c]], label = 'noise')
                else:
                    plt.scatter([X_tsne[point_to_index[tuple(p)], 0] for p in extracted_clusters[c]], [X_tsne[point_to_index[tuple(p)], 1] for p in extracted_clusters[c]])
                    
            plt.legend()
            plt.title("Spectral")
            plt.show()  
    
    return (ARI, NMI)



# iris = datasets.load_iris()    
# points = list(iris.data)
# labels = list(iris.target)
# Npts = 3
# K = 3
# M = 35 
# D = 20 
# A_init = 1 
# A_step = 0.5 
# A_max_iter = 20
# expansion_Npts = 2


# breast = datasets.load_breast_cancer()
# points = list(breast.data)
# labels = list(breast.target)
# Npts = 3
# K = 2
# M = 100 
# D = 20 
# A_init = 1 
# A_step = 0.5 
# A_max_iter = 20
# expansion_Npts = 2


digits = datasets.load_digits()    
points = list(digits.data)
labels = list(digits.target)
Npts = 3
K = 10
M = 100 
D = 1.5 
A_init = 1 
A_step = 0.5 
A_max_iter = 20
expansion_Npts = 2


# wine = datasets.load_wine()
# points = list(wine.data)
# labels = list(wine.target)


# seeds_data = fetch_openml("seeds", version=1)  
# points = np.array(seeds_data.data)
# labels = list(seeds_data.target)


# AP_breast_colon_data = fetch_openml("AP_Breast_Colon", version=1)  
# points = np.array(AP_breast_colon_data.data)
# labels = list(AP_breast_colon_data.target)


# data = fetch_openml("letter", version=1)  
# points = np.array(data.data)[16000::]
# labels = list(data.target)[16000::]


# data = fetch_openml("mnist_784", version=1)  
# points = np.array(data.data)[60000::] #[68000::] #[60000::]
# labels = list(data.target)[60000::] #[68000::] #[60000::]
# points_OG = points.copy()
# points = TSNE(n_components=2).fit_transform(points)







## Scikit learn datasets

# n_samples = 500
# seed = 30
# noisy_circles = datasets.make_circles(
#     n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
# )
# Xd, yd = noisy_circles
# points = list(Xd)
# labels = list(yd)




# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
# Xd, yd = noisy_moons
# points = list(Xd)
# labels = list(yd)


# blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
# Xd, yd = blobs
# points = list(Xd)
# labels = list(yd)


# # rng = np.random.RandomState(seed)
# # no_structure = rng.rand(n_samples, 2), None


# # Anisotropicly distributed data
# random_state = 170
# X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
# transformation = [[0.6, -0.6], [-0.4, 0.8]]
# X_aniso = np.dot(X, transformation)
# aniso = (X_aniso, y)
# Xd, yd = aniso
# points = list(Xd)
# labels = list(yd)


# # # blobs with varied variances
# varied = datasets.make_blobs(
#     n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
# )
# Xd, yd = varied
# points = list(Xd)
# labels = list(yd)




disp_2feat = 0
disp_TSNE = 1

X = np.array(points)

n = len(points)

true_clusters = {}
for i in range(n):
    if labels[i] not in true_clusters:
        true_clusters[labels[i]] = [points[i]]
    else:
        true_clusters[labels[i]].append(points[i])


point_to_index = {}
for i in range(n):
    point_to_index[tuple(points[i])] = i



(ARI, NMI) = plot_MSE(X, labels, Npts, K, M, D, A_init, A_step, A_max_iter, expansion_Npts, disp_2feat, disp_TSNE)

