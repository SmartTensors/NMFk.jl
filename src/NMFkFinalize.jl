import Statistics

"Finalize the NMFk results"
function finalize(Wa::Vector, idx::Matrix)
	nNMF = length(Wa)
	nk, nP = size(Wa[1]) # number of observation points (samples)
	nT = nk * nNMF # total number of signals to cluster

	idx_r = vec(reshape(idx, nT, 1))
	if nk > 1
		WaDist = Distances.pairwise(Distances.CosineDist(), vcat(Wa...); dims=1)
		silhouettes = Clustering.silhouettes(idx_r, WaDist)
	end

	clustersilhouettes = Vector{Float64}(undef, nk)
	W = Array{Float64}(undef, nk, nP)
	Wvar = Array{Float64}(undef, nk, nP)
	for k = 1:nk
		if nk > 1
			idxk = findall((in)(k), idx_r)
			clustersilhouettes[k] = Statistics.mean(silhouettes[idxk])
		else
			clustersilhouettes[k] = 1
		end
		idxk2 = findall((in)(k), idx)
		idxkk = [i[1] for i in idxk2]
		ws = hcat(map((i, j)->Wa[i][j, :], 1:nNMF, idxkk)...)
		W[k, :] = Statistics.mean(ws; dims=2)
		Wvar[k, :] = Statistics.var(ws; dims=2)
	end
	return W, clustersilhouettes, Wvar
end
function finalize(Wa::Vector, Ha::Vector, idx::Matrix, clusterweights::Bool=false)
	nNMF = length(Wa)
	nP = size(Wa[1], 1) # number of observation points (samples)
	nk, nC = size(Ha[1]) # number of signals / number of observations for each point (components/transients),
	nT = nk * nNMF # total number of signals to cluster

	idx_r = vec(reshape(idx, nT, 1))
	if clusterweights
		WaDist = Distances.pairwise(Distances.CosineDist(), hcat(Wa...); dims=2)
		inanw = isnan.(WaDist)
		WaDist[inanw] .= 0
		silhouettes = reshape(Clustering.silhouettes(idx_r, WaDist), nk, nNMF)
		WaDist[inanw] .= NaN
	else
		HaDist = Distances.pairwise(Distances.CosineDist(), vcat(Ha...); dims=1)
		inanh = isnan.(HaDist)
		HaDist[inanh] .= 0
		silhouettes = reshape(Clustering.silhouettes(idx_r, HaDist), nk, nNMF)
		HaDist[inanh] .= NaN
	end
	silhouettes[isnan.(silhouettes)] .= 0
	clustersilhouettes = Array{Float64}(undef, nk, 1)
	W = Array{Float64}(undef, nP, nk)
	H = Array{Float64}(undef, nk, nC)
	Wvar = Array{Float64}(undef, nP, nk)
	Hvar = Array{Float64}(undef, nk, nC)
	for k = 1:nk
		idxk = findall((in)(k), idx)
		clustersilhouettes[k] = Statistics.mean(silhouettes[idxk])
		idxkk = [i[1] for i in idxk]
		ws = hcat(map((i, j)->Wa[i][:, j], 1:nNMF, idxkk)...)
		hs = hcat(map((i, j)->Ha[i][j, :], 1:nNMF, idxkk)...)
		H[k, :] = Statistics.mean(hs; dims=2)
		W[:, k] = Statistics.mean(ws; dims=2)
		Wvar[:, k] = Statistics.var(ws; dims=2)
		Hvar[k, :] = Statistics.var(hs; dims=2)
	end
	return W, H, clustersilhouettes, Wvar, Hvar
end
function finalize(Wa::Matrix, Ha::Matrix, nNMF::Integer, idx::Matrix, clusterweights::Bool=false)
	nP = size(Wa, 1) # number of observation points (samples)
	nC = size(Ha, 2) # number of observations for each point (components/transients)
	nT = size(Ha, 1) # total number of signals to cluster
	nk = convert(Int, nT / nNMF)

	idx_r = vec(reshape(idx, nT, 1))
	if clusterweights
		WaDist = Distances.pairwise(Distances.CosineDist(), Wa; dims=2)
		silhouettes = reshape(Clustering.silhouettes(idx_r, WaDist), nk, nNMF)
	else
		HaDist = Distances.pairwise(Distances.CosineDist(), Ha; dims=1)
		silhouettes = reshape(Clustering.silhouettes(idx_r, HaDist), nk, nNMF)
	end
	clustersilhouettes = Array{Float64}(undef, nk, 1)
	W = Array{Float64}(undef, nP, nk)
	H = Array{Float64}(undef, nk, nC)
	Wvar = Array{Float64}(undef, nP, nk)
	Hvar = Array{Float64}(undef, nk, nC)
	for k = 1:nk
		idxk = findall((in)(k), idx)
		clustersilhouettes[k] = Statistics.mean(silhouettes[idxk])
		W[:, k] = Statistics.mean(Wa[:, idxk]; dims=2)
		H[k, :] = Statistics.mean(Ha[idxk, :]; dims=1)
		Wvar[:, k] = Statistics.var(Wa[:, idxk]; dims=2)
		Hvar[k, :] = Statistics.var(Ha[idxk, :]; dims=1)
	end
	return W, H, clustersilhouettes, Wvar, Hvar
end
function finalize(Wa::Matrix, Ha::Matrix)
	W = Statistics.mean(Wa; dims=2)
	H = Statistics.mean(Ha; dims=1)
	return W, H
end
function finalize(Wa::Vector, Ha::Vector)
	W = Statistics.mean(Wa[1]; dims=2)
	H = Statistics.mean(Ha[1]; dims=1)
	return W, H
end
