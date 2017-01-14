import Distances

"Cluster NMFk solutions"
function clustersolutions(H, nNMF)
	nP = size(H, 1) # number of observations (components/transients)
	nT = size(H, 2) # number of total number of sources to cluster
	nk = convert(Int, nT / nNMF )

	centroids = H[:, 1:nk]
	idx = Array(Int, nk, nNMF)

	for clusterIt = 1:nNMF
		for globalIterID = 1:nNMF
			processesTaken = falses(nk, 1)
			centroidsTaken = falses(nk, 1)
			for currentProcessID = 1:nk
				distMatrix = ones(nk, nk) + 99
				for processID = 1:nk
					if !processesTaken[processID]
						for centroidID = 1:nk
							if !centroidsTaken[centroidID]
								distMatrix[processID, centroidID] = Distances.cosine_dist(H[:,processID + (globalIterID - 1) * nk], centroids[:,centroidID])
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
				centroids[:, centroidID] += H[:, findin(idx[:, globalIterID], centroidID) + (globalIterID - 1) * nk]
			end
		end
		centroids ./= nNMF
	end
	return idx, centroids'
end