module NMFk

import NMF
import Clustering
import Distances
import Stats
import MixMatch

function execute(X::Matrix, nNMF::Int, nk::Int; ratios::Union{Void,Array{Float32, 3}}=nothing, deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=true, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), weightinverse::Bool=false, clusterweights::Bool=true, transpose::Bool=false)
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk sources ...")
	if !quiet
		if mixmatch
			if matchwaterdeltas
				println("Using MixMatchDeltas ...")
			else
				println("Using MixMatch ...")
			end
		else
			println("Using NMF ...")
		end
	end
	nP = size(X, 1) # number of observation points
	nC = size(X, 2) # number of observed components/transients
	nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	WBig = SharedArray(Float64, nP, nNMF * nk)
	HBig = SharedArray(Float64, nNMF * nk, nRC)
	objvalue = SharedArray(Float64, nNMF)
	@sync @parallel for i = 1:nNMF
		if mixmatch
			if matchwaterdeltas
				W, H, objvalue = MixMatch.matchwaterdeltas(X, nk; random=true, maxiter=maxiter, regularizationweight=regularizationweight)
			else
				if sizeof(deltas) == 0
					W, H, objvalue = MixMatch.matchdata(X, nk; ratios=ratios, random=true, mixtures=mixtures, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse, quiet=quiet)
				else
					W, Hconc, Hdeltas, objvalue = MixMatch.matchdata(X, deltas, deltaindices, nk; random=true, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse, quiet=quiet)
					H = [Hconc Hdeltas]
				end
			end
		else
			if scale
				Xn, Xmax = MixMatch.scalematrix(X)
				if transpose
					nmf_result = NMF.nnmf(Xn', nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				else
					nmf_result = NMF.nnmf(Xn, nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				end			
				W = nmf_result.W
				H = nmf_result.H
				if transpose
					W = MixMatch.descalematrix_col(W, Xmax)
				else
					H = MixMatch.descalematrix(h, Xmax)
				end
			else
				if transpose
					nmf_result = NMF.nnmf(X', nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				else
					nmf_result = NMF.nnmf(X, nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				end
				W = nmf_result.W
				H = nmf_result.H
			end			
			#=
			# Bad normalization ... it cannot work in general
			A = diagm(1 ./ vec(sum(W, 2)))
			B = (A * W * H) \ (W * H)
			W = A * W
			H = H * B
			E = X - W * H
			@show sum(E.^2)
			=#
			objvalue[i] = nmf_result.objvalue
		end
		!quiet && println("$i: Objective function = $(objvalue[i])")
		nmfindex = nk * i
		WBig[1:nP, nmfindex-(nk-1):nmfindex] = W
		HBig[nmfindex-(nk-1):nmfindex, 1:nRC] = H
	end
	!quiet && println("Best objective function = $(minimum(objvalue))")
	nmfindex = nk * indmin(objvalue)
	Wbest = WBig[1:nP, nmfindex-(nk-1):nmfindex]
	Hbest = HBig[nmfindex-(nk-1):nmfindex, 1:nRC]
	minsilhouette = 1
	if nk > 1
		# use improved k-means clustering accounting for the expected number of samples in each cluster
		# each cluster should have nNMF / nk sources!
		if clusterweights
			clusterassignments, M = NMFk.clustersolutions(WBig, nNMF) # cluster based on mixers
		else
			clusterassignments, M = NMFk.clustersolutions(HBig', nNMF) # cluster based on bucket concentrations
		end
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
