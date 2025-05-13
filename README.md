# Minimal-Seed-Expansion
The clustering algorithm introduced in the paper "Guaranteed Recovery of Unambiguous Clusters" available at:  https://arxiv.org/abs/2501.13093  

In specific, this is the algorithm used in the experiments section of the paper.  This implementation approximates the minimum A value that gives a K-clustering by progressively changing A until a K-clustering is obtained.

Npts, K, M, D are all defined in the paper.

A_init - the initial value of A tried

A_step - the step size of A used in the inital search

A_max_iter - the maximum number of A values that can be tried

expansion_Npts - the Npts value used in the greedy expansion step of the algoirhtm


If the current parameter selection doesn't yield a K-clustering, a different setting of the paramaters may work.


If you use the code provided here, please cite the paper:

Kayvon Mazooji, Ilan Shomorony, "Guaranteed Recovery of Unambiguous Clusters," arXiv, 2025, arXiv:2501.13093.
(accepted to 2025 IEEE International Symposium on Information Theory (ISIT), Ann Arbor, Michigan, June 2025)
