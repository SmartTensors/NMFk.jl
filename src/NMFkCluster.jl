import Distances

"Cluster NMFk solutions"
function clustersolutions(W::Vector, clusterweights::Bool)
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
function clustersolutions(W::Matrix, nNMF::Integer)
	nP, nT = size(W) # number of observations (components/transients), number of total number of sources to cluster
	nk = convert(Int, nT / nNMF )

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