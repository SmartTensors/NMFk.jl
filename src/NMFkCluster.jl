import Distances
import Suppressor

"""
Given a vector of classifiers, return a vector where
the highest count is now classifier #1, and the lowest count is
classifer #N

params:
	assignments (Clustering.assignments): a vector containing assignments

returns:
	remapped_assignments: a vector containing "sorted" assignments
"""
function remap2count(assignments)
	# Count how many of each classifier are in the vector
	assignment_count = [(i, count(x->x==i, assignments)) for i in unique(assignments)]

	# Sort the vector by highest -> lowest count
	sort!(assignment_count; by=x->x[2], rev=true)

	# Now, the highest count classifier will be first entry in assignment_count.
	# Create a dictionary that maps "1" => "N", where N is the most populated classifier
	remapper = Dict([(assignment_count[i][1],i) for i in 1:length(unique(assignments))])

	# Finally, map the assignments to the new ordering and return
	mfunc(i) = remapper[i]
	return map(mfunc, assignments)
end

function robustkmeans(X::Array, range::AbstractRange{Int}, repeats::Int=1000; kw...)
	best_totalcost = Inf
	best_silhouette = 0
	kbest = range[1]
	local cbest
	local best_sc
	for k in range
		if k >= size(X, 2)
			@info("$k: cannot be computed (k is greater than or equal to size(X,2); $k >= $(size(X, 2)))")
			continue
		end
		local c_new
		local mean_silhouettes
		local sc
		@Suppressor.suppress begin
			c_new, mean_silhouettes, sc = robustkmeans(X, k, repeats; kw...)
		end
		@info("$k: OF: $(c_new.totalcost) Mean Silhouette: $(mean_silhouettes) Cluster Silhouettes: $(sc)")
		if best_silhouette < mean_silhouettes
			best_silhouette = mean_silhouettes
			best_totalcost = c_new.totalcost
			cbest = deepcopy(c_new)
			best_sc = copy(sc)
			kbest = k
		end
	end
	@info("Best $kbest - OF: $best_totalcost Mean Silhouette: $best_silhouette Cluster Silhouettes: $(best_sc)")
	return cbest
end

function robustkmeans(X::Array, k::Int, repeats::Int=1000; maxiter=1000, tol=1e-32, display=:none, distance=Distances.CosineDist())
	best_totalcost = Inf
	local c
	best_mean_cluster_silhouettes = Vector{Float64}(undef, k)
	mean_cluster_silhouettes = Vector{Float64}(undef, k)
	local best_mean_silhouettes
	for i = 1:repeats
		c_new = Clustering.kmeans(X, k; maxiter=maxiter, tol=tol, display=display, distance=distance)
		Xd = Distances.pairwise(Distances.CosineDist(), X; dims=2)
		silhouettes = Clustering.silhouettes(c_new, Xd)
		mean_silhouettes = mean(silhouettes)
		for i in unique(c_new.assignments)
			mean_cluster_silhouettes[i] = mean(silhouettes[c_new.assignments.==i])
		end
		if c_new.totalcost < best_totalcost
			best_mean_cluster_silhouettes = copy(mean_cluster_silhouettes)
			best_mean_silhouettes = mean_silhouettes
			best_totalcost = c_new.totalcost
			c = deepcopy(c_new)
		end
	end
	return c, best_mean_silhouettes, best_mean_cluster_silhouettes
end

function clustersolutions(factors::Vector{Matrix}, clusterWeights::Bool=false)
	if !clusterWeights
		factors = [permutedims(f) for f in factors]
	end
	# invariant: we can now assume that our matrices are n x k
	numFactors = length(factors)
	numTrials = numFactors
	r, k = size(factors[1])
	for w in factors
		@assert size(w) == (r, k)
	end
	# fix zero case
	needZeroFix = false
	for i in 1:numFactors
		factor = factors[i]
		if minimum(sum(factor; dims=1)) == 0  # if we have a zero column
			needZeroFix = true
			break
		end
	end
	if needZeroFix
		biasRow = ones(k)'
		for i in 1:numFactors
			factors[i] = vcat(factors[i], biasRow)
		end
	end
	# invariant: at this point, no factor has a column with all zeros

	centSeeds = factors[1]
	# when we label a factor column, we are going to keep a sum
	newClusterCenters = factors[1]
	# clusterLbls[a, b] = c --> factors[b][:, a] belongs to cluster c
	clusterLbls = zeros(Int, k, numTrials)
	# note: all clusterLbls should be in [1, k] upon return

	# by definition, the columns of the first solution belong to their own cluster.
	clusterLbls[:, 1] = [i for i in 1:k]

	clusterDistances = Matrix{Float64}(undef, k, k)
	for trial in 2:numTrials
		W = factors[trial]
		# clusterDistances[a, b] = c --> dist(W[:,a], centSeeds[:,b]) = c
		for centroidIdx in 1:k
			centroid = centSeeds[:, centroidIdx]
			for factorColIdx in 1:k
				clusterDistances[factorColIdx, centroidIdx] = Distances.cosine_dist(W[:, factorColIdx], centroid)
			end
		end
		clusterDistances[isnan.(clusterDistances)] .= 0
		while minimum(clusterDistances) < Inf
			# get the row and column of the smallest distance
			selectFactorCol, selectCentIdx = Tuple(CartesianIndices(clusterDistances)[argmin(clusterDistances)])
			# save that col in trial belongs to centroid's cluster
			# println("Assigned: Trial: $trial, Factor: $selectFactorCol, Centroid: $selectCentIdx")
			clusterLbls[selectFactorCol, trial] = selectCentIdx
			# this factor cannot belong to other centSeeds
			clusterDistances[selectFactorCol, :] .+= Inf
			# this cluster cannot collect more factor columns
			clusterDistances[:, selectCentIdx] .+= Inf
			newClusterCenters[:, selectCentIdx] .+= W[:, selectFactorCol]
		end
	end
	while minimum(clusterLbls) == 0
		idx, trial = Tuple(CartesianIndices(clusterLbls)[argmin(clusterLbls)])
		if sum(clusterLbls[:, trial]) == 0
			@warn("Solution $trial was not assigned to any of the cluster!")
			clusterLbls[:, trial] = [i for i in 1:k]
		else
			@warn("Parameter $idx in solution $trial was not assigned a cluster!")
			clusterLbls[idx, trial] = idx
		end
	end
	if minimum(clusterLbls) <= 0
		@warn("Minimum assignments should be greater than 1: $(minimum(clusterLbls))")
	end
	if maximum(clusterLbls) > k
		@warn("Maximum assignments should be less than $k: $(maximum(clusterLbls))")
	end
	for i in 1:k
		for j in 1:numTrials
			l = length(findall((in)(i), clusterLbls[:, j]))
			if l != 1
				@warn("Cluster $i does not appear only once in column $j; it appears $l times!")
			end
		end
	end

	newClusterCenters ./= numTrials
	if !clusterWeights
		factors = [permutedims(f) for f in factors]
	end
	return clusterLbls, newClusterCenters'
end

"Cluster NMFk solutions"
function clustersolutions_old(W::Vector, clusterweights::Bool=false)
	nNMF = length(W)
	nc, nr = size(W[1])
	nk = clusterweights ? nr : nc

	centroids = W[1]
	idx = Array{Int}(undef, nk, nNMF)

	for clusterIt = 1:nNMF
		for globalIterID = 1:nNMF
			processesTaken = falses(nk)
			centroidsTaken = falses(nk)
			for currentProcessID = 1:nk
				distMatrix = ones(nk, nk) + Inf
				for processID = 1:nk
					if !processesTaken[processID]
						for centroidID = 1:nk
							if !centroidsTaken[centroidID]
								if clusterweights
									distMatrix[processID, centroidID] = Distances.cosine_dist(W[globalIterID][:, processID], centroids[:, centroidID])
								else
									distMatrix[processID, centroidID] = Distances.cosine_dist(W[globalIterID][processID, :], centroids[centroidID, :])
								end
							end
						end
					end
				end
				minProcess, minCentroid = Tuple(CartesianIndices(size(distMatrix))[argmin(distMatrix)])
				processesTaken[minProcess] = true
				centroidsTaken[minCentroid] = true
				idx[minProcess, globalIterID] = minCentroid
			end
		end
		centroids = zeros(size(W[1]))
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				if clusterweights
					centroids[:, centroidID] += W[globalIterID][:, findall((in)(centroidID), idx[:, globalIterID])]
				else
					centroids[centroidID:centroidID, :] += W[globalIterID][findall((in)(centroidID), idx[:, globalIterID]), :]
				end
			end
		end
		centroids ./= nNMF
	end
	return idx, centroids'
end

function clustersolutions_old(W::Matrix, nNMF::Integer)
	nP, nT = size(W) # number of observations (components/transients), number of total number of sources to cluster
	nk = convert(Int, nT / nNMF)

	centroids = W[:, 1:nk]
	idx = Array{Int}(undef, nk, nNMF)

	for clusterIt = 1:nNMF
		for globalIterID = 1:nNMF
			processesTaken = falses(nk)
			centroidsTaken = falses(nk)
			for currentProcessID = 1:nk
				distMatrix = ones(nk, nk) + Inf
				for processID = 1:nk
					if !processesTaken[processID]
						for centroidID = 1:nk
							if !centroidsTaken[centroidID]
								distMatrix[processID, centroidID] = Distances.cosine_dist(W[:,processID + (globalIterID - 1) * nk], centroids[:,centroidID])
							end
						end
					end
				end
				minProcess, minCentroid = Tuple(CartesianIndices(size(distMatrix))[argmin(distMatrix)])
				processesTaken[minProcess] = true
				centroidsTaken[minCentroid] = true
				idx[minProcess, globalIterID] = minCentroid
			end
		end
		centroids = zeros(nP, nk)
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				centroids[:, centroidID] += W[:, findall((in)(centroidID), idx[:, globalIterID]) + (globalIterID - 1) * nk]
			end
		end
		centroids ./= nNMF
	end
	return idx, centroids'
end
