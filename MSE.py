import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
import operator
import warnings 
warnings.filterwarnings("ignore")


class MSE:
    def __init__(self,
                 Npts=3,
                 K = 2,
                 M = 5,
                 D = 20,
                 A_init = 1,
                 A_step = 0.1, 
                 A_max_iter = 15,
                 expansion_Npts = 2
                 ):
                #  min_cluster_size = 5,
                #  d_thresh = 20,
                #  c_thresh = 1,
                #  c_step = 0.1,

        self.Npts = Npts
        self.K = K
        self.M = M
        self.D = D
        self.A_init = A_init
        self.A_step = A_step
        self.A_max_iter = A_max_iter
        self.expansion_Npts = expansion_Npts
        # self.min_cluster_size = min_cluster_size
        # self.d_thresh = d_thresh
        # self.c_thresh = c_thresh
        # self.c_step = c_step


        self.labels_ = None
        self.extracted_clusters = None
        self.extracted_cluster_epsilons = None
        self.init_centroids = None
        self.A_final = None


    def fit(self, X):
        print("Algorithm Progress")
        Npts = self.Npts
        K = self.K
        min_cluster_size = self.M
        d_thresh = self.D
        c_thresh = self.A_init
        c_step = self.A_step
        try_count_limit = self.A_max_iter   
        expansion_Npts = self.expansion_Npts

        points = list(X)
        n = len(points)

        point_to_index = {}
        for i in range(n):
            point_to_index[tuple(points[i])] = i

        D = sklearn.metrics.pairwise_distances(X, metric='euclidean', n_jobs=None)

        eps_for_Npts = []
        eps_for_expansion_Npts = []
        A = []
        for i in range(n):
            epsilons = list(D[i, :])
            epsilons.sort()
            A.append(epsilons)
            eps_for_Npts.append(epsilons[Npts-1])
            eps_for_expansion_Npts.append(epsilons[expansion_Npts-1])

        point_to_sorted = {}
        for i in range(n):
            point_to_dist = {}
            for j in range(n):
                if j != i:
                    point_to_dist[j] = D[i, j]    
            sorted_points = sorted(point_to_dist.items(), key=operator.itemgetter(1))
            point_to_sorted[i] = sorted_points    

        X_og = X.copy()
        D_og = D.copy()

        num_clusters = None
        c_thresh_seen = {}
        max_dist = np.max(D)
        #min_nonzero_eps_for_Npts = min([q for q in eps_for_Npts if q > 0])
        
        try_count = 0
        while num_clusters != K:
            extracted_clusters = {}
            extracted_cluster_epsilons = {}
            points_clustered = {}
            points_tried = {}
            D = D_og.copy()
            
            while True:
                min_eps = np.infty
                min_eps_inds = []   
                for i in range(n):
                    if eps_for_Npts[i] < min_eps and eps_for_Npts[i] > 0 and i not in points_clustered and i not in points_tried:
                        min_eps_inds = [i]
                        min_eps = eps_for_Npts[i]
                    elif eps_for_Npts[i] == min_eps and eps_for_Npts[i] > 0 and i not in points_clustered and i not in points_tried:
                        min_eps_inds.append(i) 

                if 0 in extracted_cluster_epsilons:
                    if c_thresh*min_eps > d_thresh*extracted_cluster_epsilons[0] or len(points_clustered) == n or min_eps == np.inf:
                        break
                else:
                    if len(points_clustered) == n or min_eps == np.inf:
                        break                
                
                if len(extracted_clusters) > K:
                    break

                cluster_eps = DBSCAN(eps = c_thresh*min_eps, min_samples = Npts, metric= 'precomputed', n_jobs = -1).fit(D)
                clusters = list(set(cluster_eps.labels_))
                cluster_eps_list = list(cluster_eps.labels_)
            
                core_points = {}
                for i in list(cluster_eps.core_sample_indices_):
                    core_points[i] = True 
                
                clusters_removed_inds = set()
                for each in min_eps_inds:
                    clusters_removed_inds.add(cluster_eps_list[each])
                    
                potential_extracted_clusters = {}
                for each in clusters_removed_inds:
                    potential_extracted_clusters[each] = set()
                for i in range(n):
                    if cluster_eps_list[i] in clusters_removed_inds and i in core_points:
                        potential_extracted_clusters[cluster_eps_list[i]].add(i)
            
                for c1 in potential_extracted_clusters:
                    cand = True
                    for c2 in extracted_clusters:
                        if len(potential_extracted_clusters[c1].intersection(extracted_clusters[c2])) > 0:
                            cand = False
                            break
                    if len(potential_extracted_clusters[c1]) < min_cluster_size:
                        cand = False
                    if cand == True:
                        extracted_clusters[len(extracted_clusters)] = potential_extracted_clusters[c1]
                        extracted_cluster_epsilons[len(extracted_cluster_epsilons)] = c_thresh*min_eps
                        for pt in potential_extracted_clusters[c1]:
                            points_clustered[pt] = True
                            D[pt, :] = 100*max_dist
                            D[:, pt] = 100*max_dist
                            D[pt, pt] = 0
                
                for pt in min_eps_inds:
                    points_tried[pt] = True
            
            num_clusters = len(extracted_clusters)
            print("A:", c_thresh, "   num_clusters:", num_clusters)
            c_thresh_seen[c_thresh] = True
            
            if num_clusters < K:
                if c_thresh > 1:
                    if max(c_thresh - c_step, 1) in c_thresh_seen:
                        c_step = c_step / 2
                        c_thresh = max(c_thresh - c_step, 1)
                    else:
                        c_thresh = max(c_thresh - c_step, 1)
                else:
                    print("Failure: pick different parameters")
                    failed_run = True
                    break        
            if num_clusters > K:
                if c_thresh + c_step in c_thresh_seen:
                    c_step = c_step / 2
                    c_thresh = c_thresh + c_step  
                else:             
                    c_thresh += c_step
            
            try_count += 1
            if try_count > try_count_limit:
                print("Failure: pick different parameters")
                print("limit for number of A values tried has been reached:", try_count_limit)
                failed_run = True
                break

        D = D_og

        labels_found = -np.ones(n)
        for c in extracted_clusters:
            for p in extracted_clusters[c]:
                labels_found[p] = c

        unclustered = {}
        init_centroids = {}
        for i in range(n):
            if i not in points_clustered:
                unclustered[i] = True
        if len(unclustered) == 0:
            unclustered = []
        else:
            closest = None
            closest_dist = np.inf
            point_to_closest = {}
            for i in unclustered:
                closest_cand = None
                closest_dist_cand = np.inf
                for j in point_to_sorted[i]:
                    if j[0] in points_clustered:
                        closest_dist_cand = j[1]
                        closest_cand = j[0]
                        break
                point_to_closest[i] = (closest_cand, closest_dist_cand)
                if closest_dist_cand < closest_dist:
                    closest_dist = closest_dist_cand
                    closest = (i, closest_cand)
            unclustered.pop(closest[0])
            extracted_clusters[labels_found[closest[1]]].add(closest[0])
            labels_found[closest[0]] = labels_found[closest[1]]
            points_clustered[closest[0]] = True

            for i in unclustered:
                if D[i, closest[0]] < point_to_closest[i][1]:
                    point_to_closest[i] = (closest[0], D[i, closest[0]])

            while len(unclustered) > 0:
                closest = None
                closest_dist = np.inf
                for i in unclustered:
                    if point_to_closest[i][1] < closest_dist:
                        closest = (i, point_to_closest[i][0])
                        closest_dist = point_to_closest[i][1]

                unclustered.pop(closest[0])
                extracted_clusters[labels_found[closest[1]]].add(closest[0])
                labels_found[closest[0]] = labels_found[closest[1]]
                points_clustered[closest[0]] = True

                for i in unclustered:
                    if D[i, closest[0]] < point_to_closest[i][1]:
                        point_to_closest[i] = (closest[0], D[i, closest[0]])
            unclustered = []

        extracted_clusters2 = {}
        for each in extracted_clusters:
            extracted_clusters2[each] = []
            for pt in extracted_clusters[each]:
                extracted_clusters2[each].append(X_og[pt, :])

        self.labels_ = labels_found
        self.extracted_clusters = extracted_clusters2
        self.extracted_cluster_epsilons = extracted_cluster_epsilons
        self.init_centroids = init_centroids
        self.A_final = c_thresh



    def fit_predict(self, X):
        self.fit(X)
        return self.labels_



