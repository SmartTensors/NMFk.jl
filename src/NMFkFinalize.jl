"Finalize the NMFk results"
function finalize(Wa::Vector, Ha::Vector, idx::Matrix, clusterweights::Bool)
	nNMF = length(Wa)
	nP = size(Wa[1], 1) # number of observation points (samples)
	nk, nC = size(Ha[1]) # number of sources / number of observations for each point (components/transients),
	nT = nk * nNMF # total number of sources to cluster

	idx_r = vec(reshape(idx, nT, 1))
	if clusterweights
		clustercounts = convert(Array{Int}, ones(nk) * nNMF)
		WaDist = Distances.pairwise(Distances.CosineDist(), hcat(Wa...))
		silhouettes = Clustering.silhouettes(idx_r, clustercounts, WaDist)
	else
		clustercounts = convert(Array{Int}, ones(nk) * nNMF)
		HaDist = Distances.pairwise(Distances.CosineDist(), vcat(Ha...)')
		silhouettes = Clustering.silhouettes(idx_r, clustercounts, HaDist)
	end
	clustersilhouettes = Array{Float64}(nk, 1)
	W = Array{Float64}(nP, nk)
	H = Array{Float64}(nk, nC)
	for k = 1:nk
		indices = findin(idx_r, k)
		clustersilhouettes[k] = mean(silhouettes[indices])
		W[:, k] = mean(hcat(map((i, j)->Wa[i][:, j], 1:nNMF, idx[k, :])...), 2)
		H[k, :] = mean(hcat(map((i, j)->Ha[i][j, :], 1:nNMF, idx[k, :])...), 2)
	end
	return W, H, clustersilhouettes
end
function finalize(Wa::Matrix, Ha::Matrix, nNMF::Integer, idx::Matrix, clusterweights::Bool)
	nP = size(Wa, 1) # number of observation points (samples)
	nC = size(Ha, 2) # number of observations for each point (components/transients)
	nT = size(Ha, 1) # total number of sources to cluster
	nk = convert(Int, nT / nNMF)

	idx_r = vec(reshape(idx, nT, 1))
	clustercounts = convert(Array{Int}, ones(nk) * nNMF)
	if clusterweights
		WaDist = Distances.pairwise(Distances.CosineDist(), Wa)
		silhouettes = Clustering.silhouettes(idx_r, clustercounts, WaDist)
	else
		HaDist = Distances.pairwise(Distances.CosineDist(), Ha')
		silhouettes = Clustering.silhouettes(idx_r, clustercounts, HaDist)
	end
	clustersilhouettes = Array{Float64}(nk, 1)
	W = Array{Float64}(nP, nk)
	H = Array{Float64}(nk, nC)
	for k = 1:nk
		indices = findin(idx_r, k)
		clustersilhouettes[k] = mean(silhouettes[indices])
		W[:, k] = mean(Wa[:, indices], 2)
		H[k, :] = mean(Ha[indices, :], 1)
	end
	return W, H, clustersilhouettes
end
function finalize(Wa::Matrix, Ha::Matrix)
	W = mean(Wa, 2)
	H = mean(Ha, 1)
	return W, H
end
function finalize(Wa::Vector, Ha::Vector)
	W = mean(Wa[1], 2)
	H = mean(Ha[1], 1)
	return W, H
end