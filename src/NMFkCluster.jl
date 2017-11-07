import Distances

function clustersolutions(factors::Vector{Matrix}, clusterWeights::Bool)
	if !clusterWeights
		factors = [f' for f in factors]
	end
	# invariant: we can now assume that our matrices are n x k

	numTrials = length(factors)
	r, k = size(factors[1])
	for w in factors
		@assert size(w) == (r, k)
	end

	# fix zero case
	needZeroFix = false
	for i in 1:length(factors)
		factor = factors[i]
		if minimum(sum(factor, 1)) == 0  # if we have a zero column
			needZeroFix = true
			break
		end
	end

	if needZeroFix
		biasRow = [1 for i in 1:k]'
		for i in 1:numTrials
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

	for trial in 2:numTrials
		W = factors[trial]
		# clusterDistances[a, b] = c --> dist(W[:,a], centSeeds[:,b]) = c
		clusterDistances = zeros(k, k) + Inf
		for centroidIdx in 1:k
			centroid = centSeeds[:, centroidIdx]
			for factorColIdx in 1:k
				column = W[:, factorColIdx]
				clusterDistances[factorColIdx, centroidIdx] = Distances.cosine_dist(column, centroid)
			end
		end

		while minimum(clusterDistances) < Inf
			# get the row and column of the smallest distance
			selectFactorCol, selectCentIdx = ind2sub(clusterDistances, indmin(clusterDistances))
			# save that col in trial belongs to centroid's cluster
			# println("Assigned: Trial: $trial, Factor: $selectFactorCol, Centroid: $selectCentIdx")
			clusterLbls[selectFactorCol, trial] = selectCentIdx
			# this factor cannot belong to other centSeeds
			clusterDistances[selectFactorCol, :] += Inf
			# this cluster cannot collect more factor columns
			clusterDistances[:, selectCentIdx] += Inf
			newClusterCenters[:, selectCentIdx] += W[:, selectFactorCol]
		end
	end

	# check our work
#  while minimum(clusterLbls) == 0
#       idx, trial = ind2sub(clusterLbls, indmin(clusterLbls))
#       println("Col $idx in trial $trial was not assigned a cluster")
#       clusterLbls[idx, trial] = k+1
#  end
	@assert minimum(clusterLbls) >= 1
	@assert maximum(clusterLbls) <= k
#  for i in 1:k, j in 1:numTrials
#    # check that cluster i appears only once in col j
#    @assert length(findin(clusterLbls[:, j], i)) == 1
#  end

	newClusterCenters ./= numTrials

	return clusterLbls, newClusterCenters'
end

"Cluster NMFk solutions"
function clustersolutions_old(W::Vector, clusterweights::Bool)
	nNMF = length(W)
	nc, nr = size(W[1])
	nk = clusterweights ? nr : nc

	centroids = W[1]
	idx = Array{Int}(nk, nNMF)

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
				minProcess, minCentroid = ind2sub(size(distMatrix), indmin(distMatrix))
				processesTaken[minProcess] = true
				centroidsTaken[minCentroid] = true
				idx[minProcess, globalIterID] = minCentroid
			end
		end
		centroids = zeros(size(W[1]))
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				if clusterweights
					centroids[:, centroidID] += W[globalIterID][:, findin(idx[:, globalIterID], centroidID)]
				else
					centroids[centroidID:centroidID, :] += W[globalIterID][findin(idx[:, globalIterID], centroidID), :]
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
	idx = Array{Int}(nk, nNMF)

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
				minProcess, minCentroid = ind2sub(size(distMatrix), indmin(distMatrix))
				processesTaken[minProcess] = true
				centroidsTaken[minCentroid] = true
				idx[minProcess, globalIterID] = minCentroid
			end
		end
		centroids = zeros(nP, nk)
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				centroids[:, centroidID] += W[:, findin(idx[:, globalIterID], centroidID) + (globalIterID - 1) * nk]
			end
		end
		centroids ./= nNMF
	end
	return idx, centroids'
end
