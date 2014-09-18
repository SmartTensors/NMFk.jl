using NMF
using Clustering

# we need to read X; this is the WL data
X = rand(5, 1000)

#TODO add a loop to perform a large number of NMF's
k = 5 #TODO add a loop to solve for k = 1, 2, 3 , 4, 5
# initialize W & H matrices
W, H = NMF.randinit(X, k, normalize=true)

# initialize W & H using Non-Negative Double Singular Value Decomposition (NNDSVD) algorithm
# Reference: C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 2007.
# W, H = NMF.nndsvd(X, k)

# Solve NMF
NMF.solve!(NMF.MultUpdate(obj=:mse,maxiter=100), X, W, H)
#NMF.solve!(NMF.ProjectedALS(maxiter=50), X, W, H)
#NMF.solve!(NMF.ALSPGrad(maxiter=50, tolg=1.0e-6), X, W, H)

# performs K-means over H, trying to group them into k clusters
# set maximum number of iterations to 200
# set display to :iter, so it shows progressive info at each iteration
R = kmeans(H, k-1; maxiter=200, display=:iter)

# the number of resultant clusters should be k
@assert nclusters(R) == k-1

# obtain the resultant assignments
# a[i] indicates which cluster the i-th sample is assigned to
a = assignments(R)

# obtain the number of samples in each cluster
# c[k] is the number of samples assigned to the k-th cluster
c = counts(R)

# get the centers (i.e. mean vectors)
# M is a matrix of size (5, 20)
# M[:,k] is the mean vector of the k-th cluster
M = R.centers

#TODO arrays to be defined based on the loops
#silhouettes(assignments(R), counts(R), R.centers)

# K-medoids is a clustering algorithm that seeks a subset of points out of a given set such that
# the total costs or distances between each point to the closest point in the chosen subset is minimal.
# This chosen subset of points are called medoids.
Q = kmedoids(W, 4; maxiter=200, display=:iter)
