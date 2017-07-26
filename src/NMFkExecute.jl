"Execute NMFk analysis for a range of number of sources"
function execute(X::Matrix, range::UnitRange{Int}, nNMF::Integer=10; kw...)
	maxsources = maximum(collect(range))
	W = Array{Array{Float64, 2}}(maxsources)
	H = Array{Array{Float64, 2}}(maxsources)
	fitquality = Array{Float64}(maxsources)
	robustness = Array{Float64}(maxsources)
	aic = Array{Float64}(maxsources)
	for numsources in range
		W[numsources], H[numsources], fitquality[numsources], robustness[numsources], aic[numsources] = NMFk.execute(X, numsources, nNMF; kw...)
	end
	return W, H, fitquality, robustness, aic
end

"Execute NMFk analysis for a given number of sources"
function execute(X::Matrix, nk::Integer, nNMF::Integer=10; casefilename::String="", serial::Bool=false, save::Bool=true, load::Bool=false, kw...)
	runflag = true
	if load && casefilename != ""
		filename = "$casefilename-$nk-$nNMF.jld"
		if isfile(filename)
			W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
			save = false
			runflag = false
		end
	end
	if runflag
		W, H, fitquality, robustness, aic = NMFk.execute_run(X, nk, nNMF; kw...)
	end
	println("Sources: $(@sprintf("%2d", nk)) Fit: $(@sprintf("%12.7g", fitquality)) Silhouette: $(@sprintf("%12.7g", robustness)) AIC: $(@sprintf("%12.7g", aic))")
	if save && casefilename != ""
		filename = "$casefilename-$nk-$nNMF.jld"
		JLD.save(filename, "W", W, "H", H, "fit", fitquality, "robustness", robustness, "aic", aic)
	end
	return W, H, fitquality, robustness, aic
end

"Execute NMFk analysis for a given number of sources in serial or parallel"
function execute_run(X::Matrix, nk::Int, nNMF::Int; acceptratio::Number=1, ipopt::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0), deltas::Matrix{Float32}=Array{Float32}(0, 0), deltaindices::Vector{Int}=Array{Int}(0), quiet::Bool=true, best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, clusterweights::Bool=true, transpose::Bool=false, sparse::Bool=false, sparsity::Number=5, sparse_cf::Symbol=:kl, sparse_div_beta::Number=-1, serial::Bool=false)
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk sources ...")
	indexnan = isnan.(X)
	if any(indexnan) && (!ipopt || !mixmatch)
		warn("The analyzed matrix has missing entries; NMF multiplex algorithm cannot be used; Ipopt minimization will be performed")
		ipopt = true
	end
	numobservations = length(vec(X[map(!, indexnan)]))
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
	# nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	if nprocs() > 1 && !serial
		r = pmap(i->(NMFk.execute_singlerun(X, nk; quiet=quiet, ipopt=ipopt, mixmatch=mixmatch, ratios=ratios, ratioindices=ratioindices, deltas=deltas, deltaindices=deltaindices, best=best, normalize=normalize, scale=scale, mixtures=mixtures, matchwaterdeltas=matchwaterdeltas, maxiter=maxiter, tol=tol, regularizationweight=regularizationweight, ratiosweight=ratiosweight, weightinverse=weightinverse, transpose=transpose, sparse=sparse, sparsity=sparsity, sparse_cf=sparse_cf, sparse_div_beta=sparse_div_beta)), 1:nNMF)
		WBig = Vector{Matrix}(nNMF)
		HBig = Vector{Matrix}(nNMF)
		for i in 1:nNMF
			WBig[i] = r[i][1]
			HBig[i] = r[i][2]
		end
		objvalue = map(i->convert(Float32, r[i][3]), 1:nNMF)
	else
		WBig = Vector{Matrix}(0)
		HBig = Vector{Matrix}(0)
		objvalue = Array{Float64}(nNMF)
		for i = 1:nNMF
			W, H, objvalue[i] = NMFk.execute_singlerun(X, nk; quiet=quiet, ipopt=ipopt, mixmatch=mixmatch, ratios=ratios, ratioindices=ratioindices, deltas=deltas, deltaindices=deltaindices, best=best, normalize=normalize, scale=scale, mixtures=mixtures, matchwaterdeltas=matchwaterdeltas, maxiter=maxiter, tol=tol, regularizationweight=regularizationweight, ratiosweight=ratiosweight, weightinverse=weightinverse, transpose=transpose, sparse=sparse, sparsity=sparsity, sparse_cf=sparse_cf, sparse_div_beta=sparse_div_beta)
			push!(WBig, W)
			push!(HBig, H)
		end
	end
	!quiet && println("Best  objective function = $(minimum(objvalue))")
	!quiet && println("Worst objective function = $(maximum(objvalue))")
	bestIdx = indmin(objvalue)
	Wbest = WBig[bestIdx]
	Hbest = HBig[bestIdx]
	if acceptratio < 1
		solind = sortperm(objvalue) .<= (nNMF * acceptratio)
		println("NNF solutions removed! Using $(sum(solind)) out of $(nNMF) solutions")
		!quiet && (println("Accepted solutions: $(objvalue[solind])"))
	else
		solind = trues(nNMF)
	end
	minsilhouette = 1
	if nk > 1
		if clusterweights
			clusterassignments, M = NMFk.clusterSolutions(WBig[solind], clusterweights) # cluster based on the W
		else
			clusterassignments, M = NMFk.clusterSolutions(HBig[solind], clusterweights) # cluster based on the sources
		end
		if !quiet
			info("Cluster assignments:")
			display(clusterassignments)
			info("Cluster centroids:")
			display(M)
		end
		Wa, Ha, clustersilhouettes = NMFk.finalize(WBig[solind], HBig[solind], clusterassignments, clusterweights)
		minsilhouette = minimum(clustersilhouettes)
		if !quiet
			info("Silhouettes for each of the $nk sources:" )
			display(clustersilhouettes')
			println("Mean silhouette = ", mean(clustersilhouettes))
			println("Min  silhouette = ", minimum(clustersilhouettes))
		end
	else
		Wa, Ha = NMFk.finalize(WBig[solind], HBig[solind])
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
		E[isnan.(E)] = 0
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
		E[isnan.(E)] = 0
		id = !isnan.(deltas)
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

"Execute single NMF run"
function execute_singlerun(x...; kw...)
	if restart
		return execute_singlerun_r3(x...; kw...)
	else
		return execute_singlerun_compute(x...; kw...)
	end
end

"Execute single NMF run without restart"
function execute_singlerun_compute(X::Matrix, nk::Int; quiet::Bool=true, ipopt::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0), deltas::Matrix{Float32}=Array{Float32}(0, 0), deltaindices::Vector{Int}=Array{Int}(0), best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, transpose::Bool=false, sparse::Bool=false, sparsity::Number=5, sparse_cf::Symbol=:kl, sparse_div_beta::Number=-1)
	if sparse
		W, H, (_, objvalue, _) = NMFk.NMFsparse(X, nk; maxiter=maxiter, tol=tol, sparsity=sparsity, cf=sparse_cf, div_beta=sparse_div_beta, quiet=quiet)
	elseif mixmatch
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
			if transpose
				Xn, Xmax = NMFk.scalematrix(X)
				Xn = Xn'
			else
				Xn, Xmax = NMFk.scalematrix(X)
			end
		else
			if transpose
				Xn = X'
			else
				Xn = X
			end
		end
		W, H = NMF.randinit(Xn, nk)
		# nmf_result = NMF.solve!(NMF.ALSPGrad{Float32}(maxiter=maxiter, tol=tol, tolg=tol*100), Xn, W, H)
		nmf_result = NMF.solve!(NMF.MultUpdate{Float32}(obj=:mse, maxiter=maxiter, tol=tol), Xn, W, H)
		!quiet && println("NMF Converged: " * string(nmf_result.converged))
		W = nmf_result.W
		H = nmf_result.H
		if scale
			if transpose
				W = NMFk.descalematrix(W, Xmax')
				E = X' - W * H
			else
				H = NMFk.descalematrix(H, Xmax)
				E = X - W * H
			end
			objvalue = sum(E.^2)
		else
			objvalue = nmf_result.objvalue
		end
	end
	!quiet && println("Objective function = $(objvalue)")
	return W, H, objvalue
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
