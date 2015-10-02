module NMFk

using NMF
using Clustering
using Distances
using Stats

export NMFrun, clustersolutions, finalize

function NMFrun(X, nk; maxiter=maxiter, normalize=false)
	W, H = NMF.randinit(X, nk, normalize = true)
	NMF.solve!(NMF.MultUpdate(obj = :mse, maxiter=maxiter), X, W, H)
	if normalize
		total = sum(W, 2)
		W ./= total
		H .*= total'
	end
	return W, H
end

function clustersolutions(H, nNMF)
	nP = size(H, 1) # number of observations (components/transients)
	nT = size(H, 2) # number of total number of sources to cluster
	nk = convert(Int, nT / nNMF )

	centroids = zeros(nP, nk);
	idx = Array(Int, nk, nNMF)

	for clusterIt = 1:nNMF
		for globalIterID = 1:nNMF
			processesTaken = zeros(nk, 1)
			centroidsTaken = zeros(nk, 1)
			for currentProcessID = 1:nk
				distMatrix = ones(nk, nk) + 99
				for processID = 1:nk
					for centroidID = 1:nk
						if centroidsTaken[centroidID] == 0 && processesTaken[processID] == 0
							distMatrix[processID, centroidID] = Distances.cosine_dist(H[:,processID + (globalIterID - 1) * nk], centroids[:,centroidID])
						end
					end
				end
				minProcess, minCentroid = ind2sub(size(distMatrix), indmin(distMatrix));
				processesTaken[minProcess] = 1;
				centroidsTaken[minCentroid] = 1;
				idx[minProcess, globalIterID] = minCentroid;
			end
		end
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				centroids[:, centroidID] += H[:, findin(idx[:, globalIterID], centroidID) + (globalIterID - 1) * nk];
			end
		end
	end
	centroids ./= nNMF
	return idx, centroids
end

function finalize(Wa, Ha, nNMF, idx)
	nC = size(Wa, 1) # number of observations (components/transients)
	nP = size(Ha, 2) # number of observation points
	nT = size(Wa, 2) # number of total number of sources to cluster
	nk = convert(Int, nT / nNMF)

	idx_r = vec(reshape(idx, nT, 1))
	clustercounts = convert(Array{Int}, ones(nk) * 10)
	WaDist = Distances.pairwise(Distances.CosineDist(), Wa)
	silhouettes = Clustering.silhouettes(idx_r, clustercounts, WaDist)
	clustersilhouettes = Array(Float64, nk, 1)
	W = Array(Float64, nC, nk)
	H = Array(Float64, nk, nP)
	for i = 1:nk
		indices = findin(idx_r, i)
		clustersilhouettes[i] = mean(silhouettes[indices])
		W[:,i] = mean(Wa[:,indices], 2)
		H[i,:] = mean(Ha[indices,:], 1)
	end
	return W, H, clustersilhouettes
end

end
