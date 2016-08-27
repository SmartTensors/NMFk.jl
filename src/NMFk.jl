module NMFk

import NMF
import Clustering
import Distances
import Stats
import MixMatch

function execute(X::Matrix, nNMF::Int, nk::Int; ratios::Union{Void,Array{Float32, 3}}=nothing, deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=true, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-12, regularizationweight::Float32=convert(Float32, 0), weightinverse::Bool=false)
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk sources ...")
	nP = size(X, 1) # number of observation points
	nC = size(X, 2) # number of observed components/transients
	WBig = Array(Float64, nP, 0)
	Wbest = Array(Float64, nP, nk)
	if sizeof(deltas) == 0
		HBig = Array(Float64, 0, nC)
		Hbest = Array(Float64, nk, nC)
	else
		numdeltas = size(deltas, 2)
		HBig = Array(Float64, 0, nC + numdeltas)
		Hbest = Array(Float64, nk, nC + numdeltas)
	end
	phi_best = Inf
	if !quiet
		if mixmatch
			if matchwaterdeltas
				println("Using MixMatchDeltas ...")
			else
				println("Using MixMatch ...")
			end
		else
			println("Using NNMF ...")
		end
	end
	for i = 1:nNMF
		if mixmatch
			if matchwaterdeltas
				W, H, objvalue = MixMatch.matchwaterdeltas(X, nk; random=true, maxiter=maxiter, regularizationweight=regularizationweight)
			else
				if sizeof(deltas) == 0
					W, H, objvalue = MixMatch.matchdata(X, nk; ratios=ratios, random=true, mixtures=mixtures, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse)
				else
					W, Hconc, Hdeltas, objvalue = MixMatch.matchdata(X, deltas, deltaindices, nk; random=true, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse)
					H = [Hconc Hdeltas]
				end
			end
		else
			nmf_result = NMF.nnmf(X, nk; alg=:multmse, maxiter=maxiter, tol=tol)
			W = nmf_result.W
			H = nmf_result.H
			objvalue = nmf_result.objvalue
		end
		!quiet && println("$i: Objective function = $objvalue")
		WBig=[WBig W]
		HBig=[HBig; H]
		if phi_best > objvalue
			phi_best = objvalue
			Wbest = W
			Hbest = H
		end
	end
	!quiet && println("Best objective function = $phi_best")
	minsilhouette = 1
	if nk > 1
		# use improved k-means clustering accounting for the expected number of samples in each cluster
		# each cluster should have nNMF / nk sources!
		clusterassignments, M = NMFk.clustersolutions(HBig', nNMF)
		!quiet && println("Cluster assignments:")
		!quiet && display(clusterassignments)
		!quiet && println("Cluster centroids:")
		!quiet && display(M)
		Wa, Ha, clustersilhouettes = NMFk.finalize(WBig, HBig, nNMF, clusterassignments)
		minsilhouette = minimum(clustersilhouettes)
		!quiet && println("Silhouettes for each of the $nk sources:" )
		!quiet && display(clustersilhouettes')
		!quiet && println("Mean silhouette = ", mean(clustersilhouettes) )
		!quiet && println("Min  silhouette = ", minimum(clustersilhouettes) )
	else
		Wa = mean(WBig, 2)
		Ha = mean(HBig, 1)
	end
	if best
		Wa = Wbest
		Ha = Hbest
	end
	if sizeof(deltas) == 0
		E = X - Wa * Ha
		E[isnan(E)] = 0
		phi_final = sum(E.^2)
	else
		Ha_conc = Ha[:,1:nC]
		Ha_deltas = Ha[:,nC+1:end]
		estdeltas = MixMatch.computedeltas(Wa, Ha_conc, Ha_deltas, deltaindices)
		E = X - Wa * Ha_conc
		E[isnan(E)] = 0
        id = !isnan(deltas)
		phi_final = sum(E.^2) + sum((deltas[id] .- estdeltas[id]).^2)
	end
	!quiet && println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )
	return Wa, Ha, phi_final, minsilhouette
end

function NMFrun(X, nk; maxiter=maxiter, normalize=true)
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

function finalize(Wa, Ha, nNMF, idx)
	nP = size(Wa, 1) # number of observation points (samples)
	nC = size(Ha, 2) # number of observations for each point (components/transients)
	nT = size(Ha, 1) # number of total number of sources to cluster
	nk = convert(Int, nT / nNMF)

	idx_r = vec(reshape(idx, nT, 1))
	clustercounts = convert(Array{Int}, ones(nk) * nNMF)
	WaDist = Distances.pairwise(Distances.CosineDist(), Wa)
	silhouettes = Clustering.silhouettes(idx_r, clustercounts, WaDist)
	clustersilhouettes = Array(Float64, nk, 1)
	W = Array(Float64, nP, nk)
	H = Array(Float64, nk, nC)
	for i = 1:nk
		indices = findin(idx_r, i)
		clustersilhouettes[i] = mean(silhouettes[indices])
		W[:,i] = mean(Wa[:,indices], 2)
		H[i,:] = mean(Ha[indices,:], 1)
	end
	return W, H, clustersilhouettes
end

end
