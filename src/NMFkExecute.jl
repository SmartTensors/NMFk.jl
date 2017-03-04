"Execute NMFk analysis"
function execute_serial(X::Matrix, nk::Int, nNMF::Int; ipopt::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array(Int, 0, 0), deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, clusterweights::Bool=true, transpose::Bool=false)
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk sources ...")
	indexnan = isnan(X)
	if any(indexnan) && (!ipopt || !mixmatch)
		warn("The analyzed matrix has missing entries; NMF multiplex algorithm cannot be used; Ipopt minimization will be performed")
		ipopt = true
	end
	numobservations = length(vec(X[!indexnan]))
	if !quiet
		if mixmatch
			if matchwaterdeltas
				println("Using MixMatchDeltas ...")
			else
				println("Using MixMatch ...")
			end
		elseif ipopt
			println("Using Ipopt ...")
		else
			println("Using NMF ...")
		end
	end
	if transpose
		nC, nP = size(X) # number of observed components/transients, number of observation points
	else
		nP, nC = size(X) # number of observation points,  number of observed components/transients
	end
	nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	WBig = Array(Float64, nP, nNMF * nk)
	HBig = Array(Float64, nNMF * nk, nRC)
	objvalue = Array(Float64, nNMF)
	for i = 1:nNMF
		W, H, objvalue[i] = execute_singlerun(X, nk; quiet=quiet, ipopt=ipopt, mixmatch=mixmatch, ratios=ratios, ratioindices=ratioindices, deltas=deltas, deltaindices=deltaindices, best=best, normalize=normalize, scale=scale, mixtures=mixtures, matchwaterdeltas=matchwaterdeltas, maxiter=maxiter, tol=tol, regularizationweight=regularizationweight, ratiosweight=ratiosweight, weightinverse=weightinverse, transpose=transpose)
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
			clusterassignments, M = NMFk.clustersolutions(WBig, nNMF) # cluster based on the W
		else
			clusterassignments, M = NMFk.clustersolutions(HBig', nNMF) # cluster based on the sources
		end
		!quiet && info("Cluster assignments:")
		!quiet && display(clusterassignments)
		!quiet && info("Cluster centroids:")
		!quiet && display(M)
		Wa, Ha, clustersilhouettes = NMFk.finalize(WBig, HBig, nNMF, clusterassignments, clusterweights)
		minsilhouette = minimum(clustersilhouettes)
		!quiet && info("Silhouettes for each of the $nk sources:" )
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
	numparameters = *(collect(size(Wa))...) + *(collect(size(Ha))...)
	if mixmatch && mixtures
		numparameters -= size(Wa)[1]
	end
	# numparameters = numbuckets # this is wrong
	# dof = numobservations - numparameters # this is correct, but we cannot use because we may get negative DoF
	# dof = maximum(range) - numparameters + 1 # this is a hack to make the dof positive.
	# dof = dof < 0 ? 0 : dof
	# sml = dof + numobservations * (log(fitquality[numbuckets]/dof) / 2 + 1.837877)
	# aic[numbuckets] = sml + 2 * numparameters
	aic = 2 * numparameters + numobservations * log(phi_final/numobservations)
	!quiet && println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )
	return Wa, Ha, phi_final, minsilhouette, aic
end

"Execute NMFk analysis (in parallel if processors are available)"
function execute(X::Matrix, nk::Int, nNMF::Int; ipopt::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array(Int, 0, 0), deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, clusterweights::Bool=true, transpose::Bool=false)
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk sources ...")
	indexnan = isnan(X)
	if any(indexnan) && (!ipopt || !mixmatch)
		warn("The analyzed matrix has missing entries; NMF multiplex algorithm cannot be used; Ipopt minimization will be performed")
		ipopt = true
	end
	numobservations = length(vec(X[!indexnan]))
	if !quiet
		if mixmatch
			if matchwaterdeltas
				println("Using MixMatchDeltas ...")
			else
				println("Using MixMatch ...")
			end
		elseif ipopt
			println("Using Ipopt ...")
		else
			println("Using NMF ...")
		end
	end
	if transpose
		nC, nP = size(X) # number of observed components/transients, number of observation points
	else
		nP, nC = size(X) # number of observation points,  number of observed components/transients
	end
	nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	r = pmap(i->(execute_singlerun(X, nk; quiet=quiet, ipopt=ipopt, mixmatch=mixmatch, ratios=ratios, ratioindices=ratioindices, deltas=deltas, deltaindices=deltaindices, best=best, normalize=normalize, scale=scale, mixtures=mixtures, matchwaterdeltas=matchwaterdeltas, maxiter=maxiter, tol=tol, regularizationweight=regularizationweight, ratiosweight=ratiosweight, weightinverse=weightinverse, transpose=transpose)), 1:nNMF)
	WBig = map(i->convert(Array{Float64,2}, r[i][1]), 1:nNMF)
	HBig = map(i->convert(Array{Float64,2}, r[i][2]), 1:nNMF)
	objvalue = map(i->convert(Float32, r[i][3]), 1:nNMF)
	bestindex = indmin(objvalue)
	!quiet && println("Best objective function = $(objvalue[bestindex])")
	Wbest = WBig[bestindex]
	Hbest = HBig[bestindex]
	minsilhouette = 1
	if nk > 1
		if clusterweights
			clusterassignments, M = NMFk.clustersolutions(WBig, clusterweights) # cluster based on the W
		else
			clusterassignments, M = NMFk.clustersolutions(HBig, clusterweights) # cluster based on the sources
		end
		!quiet && info("Cluster assignments:")
		!quiet && display(clusterassignments)
		!quiet && info("Cluster centroids:")
		!quiet && display(M)
		Wa, Ha, clustersilhouettes = NMFk.finalize(WBig, HBig, clusterassignments, clusterweights)
		minsilhouette = minimum(clustersilhouettes)
		!quiet && info("Silhouettes for each of the $nk sources:" )
		!quiet && display(clustersilhouettes')
		!quiet && println("Mean silhouette = ", mean(clustersilhouettes) )
		!quiet && println("Min  silhouette = ", minimum(clustersilhouettes) )
	else
		Wa, Ha = NMFk.finalize(WBig, HBig)
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
	numparameters = *(collect(size(Wa))...) + *(collect(size(Ha))...)
	if mixmatch && mixtures
		numparameters -= size(Wa)[1]
	end
	# numparameters = numbuckets # this is wrong
	# dof = numobservations - numparameters # this is correct, but we cannot use because we may get negative DoF
	# dof = maximum(range) - numparameters + 1 # this is a hack to make the dof positive.
	# dof = dof < 0 ? 0 : dof
	# sml = dof + numobservations * (log(fitquality[numbuckets]/dof) / 2 + 1.837877)
	# aic[numbuckets] = sml + 2 * numparameters
	aic = 2 * numparameters + numobservations * log(phi_final/numobservations)
	!quiet && println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )
	return Wa, Ha, phi_final, minsilhouette, aic
end

function execute_singlerun(X::Matrix, nk::Int; ipopt::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array(Int, 0, 0), deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, transpose::Bool=false)
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
	elseif ipopt
		W, H, objvalue = NMFk.ipopt(X, nk; random=true, normalize=normalize, scale=scale, maxiter=maxiter, regularizationweight=regularizationweight, weightinverse=weightinverse, quiet=quiet)
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
			objvalue = sum(E.^2)
		else
			if transpose
				nmf_result = NMF.nnmf(X', nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
			else
				nmf_result = NMF.nnmf(X, nk; alg=:alspgrad, init=:random, maxiter=maxiter, tol=tol)
			end
			W = nmf_result.W
			H = nmf_result.H
			objvalue = nmf_result.objvalue
		end
	end
	!quiet && println("Objective function = $(objvalue)")
	return W, H, objvalue
end

function execute(X::Matrix, range::Union{UnitRange{Int},Int}=2; retries::Integer=10, ipopt::Bool=false, quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), weightinverse::Bool=false, clusterweights::Bool=true, transpose::Bool=false, casefilename::String="")
	maxsources = maximum(collect(range))
	W = Array(Array{Float64, 2}, maxsources)
	H = Array(Array{Float64, 2}, maxsources)
	fitquality = Array(Float64, maxsources)
	robustness = Array(Float64, maxsources)
	aic = Array(Float64, maxsources)
	for numsources in range
		W[numsources], H[numsources], fitquality[numsources], robustness[numsources], aic[numsources] = NMFk.execute(X, numsources, retries;  mixmatch=mixmatch, normalize=normalize, scale=scale, mixtures=mixtures, quiet=quiet, regularizationweight=regularizationweight, weightinverse=weightinverse, clusterweights=clusterweights, transpose=transpose)
		println("Sources: $(@sprintf("%2d", numsources)) Fit: $(@sprintf("%12.7g", fitquality[numsources])) Silhouette: $(@sprintf("%12.7g", robustness[numsources])) AIC: $(@sprintf("%12.7g", aic[numsources]))")
		if casefilename != ""
			filename = "casefilename-$numsources-$retries.jld"
			JLD.save("W", W[numsources], "H", H[numsources], "fit", fitquality[numsources], "robustness", robustness[numsources], "aic", aic[numsources], "regularizationweight", regularizationweight)
		end
	end
	return W, H, fitquality, robustness, aic
end

function NMFrun(X::Matrix, nk::Integer; maxiter::Integer=maxiter, normalize::Bool=true)
	W, H = NMF.randinit(X, nk, normalize = true)
	NMF.solve!(NMF.MultUpdate(obj = :mse, maxiter=maxiter), X, W, H)
	if normalize
		total = sum(W, 2)
		W ./= total
		H .*= total'
	end
	return W, H
end