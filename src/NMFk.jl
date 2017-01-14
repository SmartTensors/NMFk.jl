module NMFk

import NMF
import Distances
import Clustering
import JuMP
import Ipopt

include("NMFkCluster.jl")
include("NMFkGeoChem.jl")
include("NMFkMixMatch.jl")
include("NMFkMatrix.jl")

"Execute NMFk analysis (in parallel if processors available)"
function execute(X::Matrix, nNMF::Int, nk::Int; ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array(Int, 0, 0), deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, clusterweights::Bool=true, transpose::Bool=false)
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
	if transpose
		nP = size(X, 2) # number of observation points
		nC = size(X, 1) # number of observed components/transients
	else
		nP = size(X, 1) # number of observation points
		nC = size(X, 2) # number of observed components/transients
	end
	nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	WBig = SharedArray(Float64, nP, nNMF * nk)
	HBig = SharedArray(Float64, nNMF * nk, nRC)
	objvalue = SharedArray(Float64, nNMF)
	@sync @parallel for i = 1:nNMF
		if mixmatch
			if matchwaterdeltas
				W, H, objvalue = NMFk.mixmatchwaterdeltas(X, nk; random=true, maxiter=maxiter, regularizationweight=regularizationweight)
			else
				if sizeof(deltas) == 0
					W, H, objvalue = NMFk.mixmatchdata(X, nk; ratios=ratios, ratioindices=ratioindices, random=true, mixtures=mixtures, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse, ratiosweight=ratiosweight, quiet=quiet)
				else
					W, Hconc, Hdeltas, objvalue = NMFk.mixmatchdata(X, deltas, deltaindices, nk; random=true, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse, ratiosweight=ratiosweight, quiet=quiet)
					H = [Hconc Hdeltas]
				end
			end
		else
			if scale
				Xn, Xmax = NMFk.scalematrix(X)
				if transpose
					nmf_result = NMF.nnmf(Xn', nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				else
					nmf_result = NMF.nnmf(Xn, nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				end
				W = nmf_result.W
				H = nmf_result.H
				if transpose
					W = NMFk.descalematrix_col(W, Xmax)
					E = X' - W * H
				else
					H = NMFk.descalematrix(H, Xmax)
					E = X - W * H
				end
				objvalue[i] = sum(E.^2)
			else
				if transpose
					nmf_result = NMF.nnmf(X', nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				else
					nmf_result = NMF.nnmf(X, nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
				end
				W = nmf_result.W
				H = nmf_result.H
				objvalue[i] = nmf_result.objvalue
			end
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
		if clusterweights
			clusterassignments, M = NMFk.clustersolutions(WBig, nNMF) # cluster based on the mixers
		else
			clusterassignments, M = NMFk.clustersolutions(HBig', nNMF) # cluster based on the sources
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
		if transpose
			E = X' - Wa * Ha
		else
			E = X - Wa * Ha
		end
		E[isnan(E)] = 0
		phi_final = sum(E.^2)
	else
		Ha_conc = Ha[:,1:nC]
		Ha_deltas = Ha[:,nC+1:end]
		estdeltas = NMFk.computedeltas(Wa, Ha_conc, Ha_deltas, deltaindices)
		if transpose
			E = X' - Wa * Ha_conc
		else
			E = X - Wa * Ha_conc
		end
		E[isnan(E)] = 0
		id = !isnan(deltas)
		phi_final = sum(E.^2) + sum((deltas[id] .- estdeltas[id]).^2)
	end
	if !quiet && typeof(ratios) != Void
		ratiosreconstruction = 0
		for (j, c1, c2) in zip(1:length(ratioindices[1,:]), ratioindices[1,:], ratioindices[2,:])
			for i = 1:nP
				s1 = 0
				s2 = 0
				for k = 1:nk
					s1 += Wa[i, k] * Ha[k, c1]
					s2 += Wa[i, k] * Ha[k, c2]
				end
				ratiosreconstruction += ratiosweight * (s1/s2 - ratios[i, j])^2
			end
		end
		println("Ratio reconstruction = $ratiosreconstruction")
		phi_final += ratiosreconstruction
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

"Finalize the NMFk results"
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