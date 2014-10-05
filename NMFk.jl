using NMF
using Clustering
# using Gadfly # Gadly demages Clustering
using Wells

dd = Wells.solve( Wells.WellsD, Wells.WellsQ, Wells.Points, Wells.time, Wells.T, Wells.S )
println(keys(dd))
numrows = size(collect(keys(dd)))[1]
numcols = size(dd[collect(keys(dd))[1]])[1]
X = Array(Float64, numrows, numcols)
k = 0
for i in keys(dd)
	k += 1
	println(i)
	# println(dd[i])
	X[k,:] = dd[i]
	# p = plot(x=Wells.time, y=dd[i], Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Geom.line )
	# draw(PNG(string("nmfk-test-20141005-",i,".png"), 6inch, 3inch), p)
end
println("Size of the matrix to solve ",size(X))
# writecsv("nmfk-test-20141005-well-names.csv",collect(keys(dd))')
# writecsv("nmfk-test-20141005.csv",X')

# RANDOM test
# X = rand(5, 1000)

k = 6 # TODO add a loop to solve for k = 1, 2, 3 , 4, 5
nNMF = 100
WBig = Array(Float64, numrows, 0)
HBig = Array(Float64, 0, numcols)
phi = Array(Float64, nNMF)
for n = 1:nNMF
	println("NMF ", n)
	# initialize W & H matrices
	W, H = NMF.randinit(X, k, normalize=true)
	# println("Size of W = ", size(W) )
	# println("Size of H = ", size(H) )

	# initialize W & H using Non-Negative Double Singular Value Decomposition (NNDSVD) algorithm
	# Reference: C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 2007.
	# W, H = NMF.nndsvd(X, k)

	# Solve NMF
	NMF.solve!(NMF.MultUpdate(obj=:mse,maxiter=100), X, W, H)
	# NMF.solve!(NMF.ProjectedALS(maxiter=50), X, W, H)
	# NMF.solve!(NMF.ALSPGrad(maxiter=50, tolg=1.0e-6), X, W, H)
	E = X - W * H
	phi[n] = sum( E' * E )
	println("Objective function = ", phi[n], " Max error = ", maximum(E), " Min error = ", minimum(E) )
	# println("Size of W = ", size(W) )
	# println("Size of WBig = ", size(WBig) )
	# println("Size of HBig = ", size(HBig) )
	WBig=[WBig W]
	HBig=[HBig, H]
	# println(WBig)
	# println(W)
end
println("NMF done.")
println("Size of WBig = ", size(WBig) )
println("Size of HBig = ", size(HBig) )
# println(WBig)

# performs K-means over W, trying to group them into k clusters
# set maximum number of iterations to 200
# set display to :iter, so it shows progressive info at each iteration
R = kmeans(HBig', k; maxiter=200, display=:iter)

# the number of resultant clusters should be k
@assert nclusters(R) == k

# Cluster assignments
# a[i] indicates which cluster the i-th sample is assigned to
clusterassignments = assignments(R)
# println(a)

# Number of samples in each cluster
# c[k] is the number of samples assigned to the k-th cluster
clustercounts = counts(R)
println("Number of samples in eash cluster = ", clustercounts)

# Cluster centers (i.e. mean vectors)
# M is a matrix of size (numcols, k)
# M[:,k] is the mean vector of the k-th cluster
M = R.centers
println("Size of matrix contamining cluster centers = ", size(M))
# println("Cluser centers = ", M)

# Silhouettes
s = silhouettes(clusterassignments, clustercounts, HBig*HBig')
println("Silhouettes vector size = ", size(s))
println("Silhouettes Avg = ", sum(s)/size(s)[1], " Max = ", maximum(s), " Min = ", minimum(s) )
# println("Silhouettes vector = ", s)

# K-medoids is a clustering algorithm that seeks a subset of points out of a given set such that
# the total costs or distances between each point to the closest point in the chosen subset is minimal.
# This chosen subset of points are called medoids.
# Q = kmedoids(W, 4; maxiter=200, display=:iter)
