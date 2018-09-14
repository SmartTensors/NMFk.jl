"Test NMFk functions"
function test()
	include(joinpath(nmfkdir, "test", "runtests.jl"))
end

"Execute NMFk analysis for a range of number of sources"
function execute(X::Array{T,N}, range::Range{Int}, nNMF::Integer=10; kw...) where {T, N}
	maxk = maximum(collect(range))
	W = Array{Array{T, N}}(maxk)
	H = Array{Array{T, 2}}(maxk)
	fitquality = Array{T}(maxk)
	robustness = Array{T}(maxk)
	aic = Array{T}(maxk)
	for nk in range
		W[nk], H[nk], fitquality[nk], robustness[nk], aic[nk] = NMFk.execute(X, nk, nNMF; kw...)
	end
	info("Results")
	for nk in range
		println("Signals: $(@sprintf("%2d", nk)) Fit: $(@sprintf("%12.7g", fitquality[nk])) Silhouette: $(@sprintf("%12.7g", robustness[nk])) AIC: $(@sprintf("%12.7g", aic[nk]))")
	end
	return W, H, fitquality, robustness, aic
end

"Execute NMFk analysis for a given number of sources"
function execute(X::Union{Matrix,Array}, nk::Integer, nNMF::Integer=10; resultdir::AbstractString=".", casefilename::AbstractString="", save::Bool=true, load::Bool=false, kw...)
	runflag = true
	if load && casefilename != ""
		filename = joinpath(resultdir, "$casefilename-$nk-$nNMF.jld")
		if isfile(filename)
			W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
			save = false
			runflag = false
		else
			info("File $filename is missing; runs will be executed!")
		end
	end
	if runflag
		W, H, fitquality, robustness, aic = NMFk.execute_run(X, nk, nNMF; casefilename=casefilename, kw...)
	end
	println("Signals: $(@sprintf("%2d", nk)) Fit: $(@sprintf("%12.7g", fitquality)) Silhouette: $(@sprintf("%12.7g", robustness)) AIC: $(@sprintf("%12.7g", aic))")
	if save && casefilename != ""
		filename = joinpath(resultdir, "$casefilename-$nk-$nNMF.jld")
		if !isdir(resultdir)
			mkdir(resultdir)
		end
		JLD.save(filename, "W", W, "H", H, "fit", fitquality, "robustness", robustness, "aic", aic)
	end
	return W, H, fitquality, robustness, aic
end

"Execute NMFk analysis for a given number of sources in serial or parallel"
function execute_run(X::Array, nk::Int, nNMF::Int; clusterweights::Bool=false, acceptratio::Number=1, acceptfactor::Number=Inf, quiet::Bool=NMFk.quiet, best::Bool=true, serial::Bool=false, method::Symbol=:nmf, algorithm::Symbol=:multdiv, zeronans::Bool=true, casefilename::AbstractString="", loadall::Bool=false, saveall::Bool=false, kw...)
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk signals (sources) ...")
	indexnan = isnan.(X)
	runflag = true
	if loadall && casefilename != ""
		filename = "$casefilename-$nk-$nNMF-all.jld"
		if isfile(filename)
			WBig, HBig, objvalue = JLD.load(filename, "W", "H", "fit")
			saveall = false
			runflag = false
		end
	end
	if runflag
		if nprocs() > 1 && !serial
			kw_dict = Dict()
			for (key, value) in kw
				kw_dict[key] = value
			end
			if haskey(kw_dict, :seed)
				kwseed = kw_dict[:seed]
				delete!(kw_dict, :seed)
				r = pmap(i->(NMFk.execute_singlerun(X, nk; quiet=true, seed=kwseed+i, kw_dict...)), 1:nNMF)
			else
				r = pmap(i->(NMFk.execute_singlerun(X, nk; quiet=true, kw...)), 1:nNMF)
			end
			WBig = Vector{Array}(nNMF)
			HBig = Vector{Matrix}(nNMF)
			for i in 1:nNMF
				WBig[i] = r[i][1]
				HBig[i] = r[i][2]
			end
			objvalue = map(i->convert(Float32, r[i][3]), 1:nNMF)
		else
			WBig = Vector{Array}(nNMF)
			HBig = Vector{Matrix}(nNMF)
			objvalue = Array{Float64}(nNMF)
			kw_dict = Dict()
			for (key, value) in kw
				kw_dict[key] = value
			end
			if haskey(kw_dict, :seed)
				kwseed = kw_dict[:seed]
				delete!(kw_dict, :seed)
				for i = 1:nNMF
					!quiet && info("NMF run #$(i)")
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; quiet=quiet, seed=kwseed+i, kw_dict...)
				end
			else
				for i = 1:nNMF
					!quiet && info("NMF run #$(i)")
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; quiet=quiet, kw...)
				end
			end
		end
	end
	idxsort = sortperm(objvalue)
	bestIdx = idxsort[1]
	!quiet && println("Best  objective function = $(objvalue[bestIdx])")
	!quiet && println("Worst objective function = $(objvalue[idxsort[end]])")
	Wbest = copy(WBig[bestIdx])
	Hbest = copy(HBig[bestIdx])
	println()
	if acceptratio < 1
		ccc = convert(Int, (ceil(nNMF * acceptratio)))
		idxrat = vec([trues(ccc); falses(nNMF-ccc)])
		println("NMF solutions removed based on an acceptance ratio: $(sum(idxrat)) out of $(nNMF) solutions remain")
	else
		idxrat = trues(nNMF)
	end
	if acceptfactor < Inf
		cutoff = objvalue[bestIdx] * acceptfactor
		idxcut = objvalue[idxsort] .< cutoff
		println("NMF solutions removed based on an acceptance factor: $(sum(idxcut)) out of $(nNMF) solutions remain")
	else
		idxcut = trues(nNMF)
	end
	idxnan = trues(nNMF)
	for i in 1:nNMF
		if sum(isnan.(WBig[i])) > 0
			idxnan[idxsort[i]] = false
		elseif sum(isnan.(HBig[i])) > 0
			idxnan[idxsort[i]] = false
		end
	end
	if sum(idxnan) < nNMF
		println("NMF solutions removed because they contain NaN's: $(sum(idxnan)) out of $(nNMF) solutions remain")
	end
	idxsol = idxrat .& idxcut .& idxnan
	if sum(idxsol) < nNMF
		println("NMF solutions removed based on acceptance criteria: $(sum(idxsol)) out of $(nNMF) solutions remain")
		println("OF: min $(minimum(objvalue)) max $(maximum(objvalue)) mean $(mean(objvalue)) std $(std(objvalue)) (ALL)")
	end
	println("OF: min $(minimum(objvalue[idxsol])) max $(maximum(objvalue[idxsol])) mean $(mean(objvalue[idxsol])) std $(std(objvalue[idxsol]))")
	Xe = NMFk.mixmatchcompute(X, Wbest, Hbest)
	minsilhouette = 1
	if nk > 1
		clusterweights = false
		clusterassignments, M = NMFk.clustersolutions(HBig[idxsort][idxsol], clusterweights) # cluster based on the sources
		if !quiet
			info("Cluster assignments:")
			display(clusterassignments)
			info("Cluster centroids:")
			display(M)
		end
		ci = clusterassignments[:, 1]
		for (i, c) in enumerate(ci)
			Wbest[:, i] = WBig[bestIdx][:, c]
			Hbest[i, :] = HBig[bestIdx][c, :]
		end
		Wa, Ha, clustersilhouettes, Wv, Hv = NMFk.finalize(WBig[idxsort][idxsol], HBig[idxsort][idxsol], clusterassignments, clusterweights)
		minsilhouette = minimum(clustersilhouettes)
		if !quiet
			info("Silhouettes for each of the $nk clusters:" )
			display(clustersilhouettes')
			println("Mean silhouette = ", mean(clustersilhouettes))
			println("Min  silhouette = ", minimum(clustersilhouettes))
		end
	else
		Wv = NaN
		Hv = NaN
		Wa, Ha = NMFk.finalize(WBig[idxsol], HBig[idxsol])
	end
	if saveall && casefilename != ""
		filename = "$casefilename-$nk-$nNMF-all.jld"
		JLD.save(filename, "W", WBig, "H", HBig, "Wmean", Wa, "Hmean", Ha, "Wvar", Wv, "Hvar", Hv, "Wbest", Wbest, "Hbest", Hbest, "fit", objvalue)
	end
	if best
		Wa = Wbest
		Ha = Hbest
	end
	E = abs.(X .- Xe)
	E[isnan.(E)] = 0
	phi_final = sum(E.^2)
	numobservations = length(vec(X[.!indexnan]))
	numparameters = *(collect(size(Wa))...) + *(collect(size(Ha))...)
	numparameters -= (size(Wa)[1] + size(Wa)[3])
	aic = 2 * numparameters + numobservations * log(phi_final/numobservations)
	!quiet && println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E))
	return Wa, Ha, phi_final, minsilhouette, aic
end
function execute_run(X::Matrix, nk::Int, nNMF::Int; clusterweights::Bool=false, acceptratio::Number=1, acceptfactor::Number=Inf, quiet::Bool=NMFk.quiet, best::Bool=true, transpose::Bool=false, serial::Bool=false, deltas::Matrix{Float32}=Array{Float32}(0, 0), ratios::Array{Float32, 2}=Array{Float32}(0, 0), mixture::Symbol=:null, method::Symbol=:nmf, algorithm::Symbol=:multdiv, casefilename::AbstractString="", zeronans::Bool=true, loadall::Bool=false, saveall::Bool=false, kw...)
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && info("NMFk analysis of $nNMF NMF runs assuming $nk sources (signals) ...")
	indexnan = isnan.(X)
	if any(indexnan) && (method != :ipopt && method != :nlopt && mixture == :null)
		warn("The analyzed matrix has missing entries; NMF multiplex algorithm cannot be used (method=$(method)); Ipopt minimization will be performed!")
		method = :ipopt
	end
	if mixture != :null
		clusterweights = true
		if method == :nmf
			method = :ipopt
		end
	end
	if method == :nlopt && algorithm == :multdiv
		algorithm = :LD_SLSQP
	end
	if method == :multdiv
		method = :nmf
		algorithm = :multdiv
	elseif method == :multmse
		method = :nmf
		algorithm = :multmse
	elseif method == :alspgrad
		method = :nmf
		algorithm = :alspgrad
	end
	if !quiet
		if mixture == :mixmatch
			print("MixMatch using ")
		elseif mixture == :matchwaterdeltas
			print("MixMatchDeltas using ")
		end
		if method == :ipopt
			println("Ipopt ...")
		elseif method == :nlopt
			println("NLopt ...")
		elseif method == :nmf
			if algorithm == :multdiv
				println("NMF Multiplicative update using divergence ...")
			elseif algorithm == :multmse
				println("NMF Multiplicative update using mean-squared-error ...")
			elseif algorithm == :alspgrad
				println("NMF Alternate Least Square using Projected Gradient Descent ...")
			end
		elseif method == :sparse
			println("Sparse NMF ...")
		elseif method == :simple
			println("Simple NMF multiplicative ...")
		end
	end
	if transpose
		nC, nP = size(X) # number of observed components/transients, number of observation points
	else
		nP, nC = size(X) # number of observation points,  number of observed components/transients
	end
	# nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	runflag = true
	if loadall && casefilename != ""
		filename = "$casefilename-$nk-$nNMF-all.jld"
		if isfile(filename)
			WBig, HBig, objvalue = JLD.load(filename, "W", "H", "fit")
			saveall = false
			runflag = false
		end
	end
	if runflag
		if nprocs() > 1 && !serial
			kw_dict = Dict()
			for (key, value) in kw
				kw_dict[key] = value
			end
			if haskey(kw_dict, :seed)
				kwseed = kw_dict[:seed]
				delete!(kw_dict, :seed)
				r = pmap(i->(NMFk.execute_singlerun(X, nk; quiet=true, best=best, transpose=transpose, deltas=deltas, ratios=ratios, mixture=mixture, method=method, algorithm=algorithm, seed=kwseed+i, kw_dict...)), 1:nNMF)
			else
				r = pmap(i->(NMFk.execute_singlerun(X, nk; quiet=true, best=best, transpose=transpose, deltas=deltas, ratios=ratios, mixture=mixture, method=method, algorithm=algorithm, kw...)), 1:nNMF)
			end
			WBig = Vector{Matrix}(nNMF)
			HBig = Vector{Matrix}(nNMF)
			for i in 1:nNMF
				WBig[i] = r[i][1]
				HBig[i] = r[i][2]
			end
			objvalue = map(i->convert(Float32, r[i][3]), 1:nNMF)
		else
			WBig = Vector{Matrix}(nNMF)
			HBig = Vector{Matrix}(nNMF)
			objvalue = Array{Float64}(nNMF)
			kw_dict = Dict()
			for (key, value) in kw
				kw_dict[key] = value
			end
			if haskey(kw_dict, :seed)
				kwseed = kw_dict[:seed]
				delete!(kw_dict, :seed)
				for i = 1:nNMF
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; quiet=quiet, best=best, transpose=transpose, deltas=deltas, ratios=ratios, mixture=mixture, method=method, algorithm=algorithm, seed=kwseed+i, kw_dict...)
				end
			else
				for i = 1:nNMF
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; quiet=quiet, best=best, transpose=transpose, deltas=deltas, ratios=ratios, mixture=mixture, method=method, algorithm=algorithm, kw...)
				end
			end
		end
	end
	idxsort = sortperm(objvalue)
	bestIdx = idxsort[1]
	!quiet && println("Best  objective function = $(objvalue[bestIdx])")
	!quiet && println("Worst objective function = $(objvalue[idxsort[end]])")
	Wbest = copy(WBig[bestIdx])
	Hbest = copy(HBig[bestIdx])
	println()
	if acceptratio < 1
		ccc = convert(Int, (ceil(nNMF * acceptratio)))
		idxrat = vec([trues(ccc); falses(nNMF-ccc)])
		println("NMF solutions removed based on an acceptance ratio: $(sum(idxrat)) out of $(nNMF) solutions remain")
	else
		idxrat = trues(nNMF)
	end
	if acceptfactor < Inf
		cutoff = objvalue[bestIdx] * acceptfactor
		idxcut = objvalue[idxsort] .< cutoff
		println("NMF solutions removed based on an acceptance factor: $(sum(idxcut)) out of $(nNMF) solutions remain")
	else
		idxcut = trues(nNMF)
	end
	idxnan = trues(nNMF)
	for i in 1:nNMF
		isnw = isnan.(WBig[i])
		isnh = isnan.(HBig[i])
		if sum(isnw) > 0
			if zeronans
				WBig[i][isnw] .= 0
			end
			idxnan[idxsort[i]] = false
		elseif sum(isnh) > 0
			if zeronans
				HBig[i][isnh] .= 0
			end
			idxnan[idxsort[i]] = false
		end
	end
	if zeronans
		warn("NMF solutions contain NaN's: $(sum(idxnan)) out of $(nNMF) solutions! Nan's have been removed!")
		idxnan = trues(nNMF)
	end
	if sum(idxnan) < nNMF
		println("NMF solutions removed because they contain NaN's: $(sum(idxnan)) out of $(nNMF) solutions remain")
	end
	idxsol = idxrat .& idxcut .& idxnan
	if sum(idxsol) < nNMF
		println("NMF solutions removed based on acceptance criteria: $(sum(idxsol)) out of $(nNMF) solutions remain")
		println("OF: min $(minimum(objvalue)) max $(maximum(objvalue)) mean $(mean(objvalue)) std $(std(objvalue)) (ALL)")
	end
	println("OF: min $(minimum(objvalue[idxsol])) max $(maximum(objvalue[idxsol])) mean $(mean(objvalue[idxsol])) std $(std(objvalue[idxsol]))")
	Xe = Wbest * Hbest
	fn = vecnormnan(X)
	println("Worst correlation by columns: $(minimum(map(i->cornan(X[i, :], Xe[i, :]), 1:size(X, 1))))")
	println("Worst correlation by rows: $(minimum(map(i->cornan(X[:, i], Xe[:, i]), 1:size(X, 2))))")
	println("Worst norm by columns: $(maximum(map(i->(vecnormnan(X[i, :] - Xe[i, :])/fn), 1:size(X, 1))))")
	println("Worst norm by rows: $(maximum(map(i->(vecnormnan(X[:, i] - Xe[:, i])/fn), 1:size(X, 2))))")
	minsilhouette = 1
	if nk > 1
		if clusterweights
			clusterassignments, M = NMFk.clustersolutions(WBig[idxsort][idxsol], clusterweights) # cluster based on the W
		else
			clusterassignments, M = NMFk.clustersolutions(HBig[idxsort][idxsol], clusterweights) # cluster based on the sources
		end
		if !quiet
			info("Cluster assignments:")
			display(clusterassignments)
			info("Cluster centroids:")
			display(M)
		end
		ci = clusterassignments[:, 1]
		for (i, c) in enumerate(ci)
			Wbest[:, i] = WBig[bestIdx][:, c]
			Hbest[i, :] = HBig[bestIdx][c, :]
		end
		Wa, Ha, clustersilhouettes, Wv, Hv = NMFk.finalize(WBig[idxsort][idxsol], HBig[idxsort][idxsol], clusterassignments, clusterweights)
		minsilhouette = minimum(clustersilhouettes)
		if !quiet
			info("Silhouettes for each of the $nk clusters:" )
			display(clustersilhouettes')
			println("Mean silhouette = ", mean(clustersilhouettes))
			println("Min  silhouette = ", minimum(clustersilhouettes))
		end
	else
		Wv = NaN
		Hv = NaN
		Wa, Ha = NMFk.finalize(WBig[idxsol], HBig[idxsol])
	end
	if saveall && casefilename != ""
		filename = "$casefilename-$nk-$nNMF-all.jld"
		JLD.save(filename, "W", WBig, "H", HBig, "Wmean", Wa, "Hmean", Ha, "Wvar", Wv, "Hvar", Hv, "Wbest", Wbest, "Hbest", Hbest, "fit", objvalue)
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
	if !quiet && sizeof(ratios) > 0
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
	numobservations = length(vec(X[.!indexnan]))
	numparameters = *(collect(size(Wa))...) + *(collect(size(Ha))...)
	if mixture != :null
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

"Execute single NTF run without restart"
function execute_singlerun_compute(X::Array, nk::Int; kw...)
	NMFk.mixmatchdata(X, nk; kw...)
end

"Execute single NMF run without restart"
function execute_singlerun_compute(X::Matrix, nk::Int; quiet::Bool=NMFk.quiet, ratios::Array{Float32, 2}=Array{Float32}(0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0), deltas::Matrix{Float32}=Array{Float32}(0, 0), deltaindices::Vector{Int}=Array{Int}(0), best::Bool=true, normalize::Bool=false, scale::Bool=false, maxiter::Int=10000, tol::Float64=1e-19, ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, transpose::Bool=false, mixture::Symbol=:null, method::Symbol=:nmf, algorithm::Symbol=:multdiv, clusterweights::Bool=false, bootstrap::Bool=false, kw...)
	if scale
		if transpose
			Xn, Xmax = NMFk.scalematrix!(X)
			Xn = Xn'
		else
			Xn, Xmax = NMFk.scalematrix!(X)
		end
	else
		if transpose
			Xn = X'
		else
			Xn = X
		end
	end
	if bootstrap
		Xn = bootstrapping(Xn)
	end
	if mixture != :null
		if mixture == :mixmatch
			if sizeof(deltas) == 0
				W, H, objvalue = NMFk.mixmatchdata(Xn, nk; method=method, algorithm=algorithm, ratios=ratios, ratioindices=ratioindices, normalize=normalize, scale=false, maxiter=maxiter, weightinverse=weightinverse, ratiosweight=ratiosweight, quiet=quiet, tol=tol, kw...)
			else
				W, Hconc, Hdeltas, objvalue = NMFk.mixmatchdeltas(Xn, deltas, deltaindices, nk; method=method, algorithm=algorithm, normalize=normalize, scale=false, maxiter=maxiter, weightinverse=weightinverse, ratiosweight=ratiosweight, quiet=quiet, tol=tol, kw...)
				H = [Hconc Hdeltas]
			end
		elseif mixture == :matchwaterdeltas
			W, H, objvalue = NMFk.mixmatchwaterdeltas(Xn, nk; method=method, algorithm=algorithm, tol=tol, maxiter=maxiter, kw...)
		end
	elseif method == :sparse
		W, H, (_, objvalue, _) = NMFk.NMFsparse(Xn, nk; maxiter=maxiter, tol=tol, quiet=quiet, kw...)
	elseif method == :ipopt || method == :nlopt
		W, H, objvalue = NMFk.jump(Xn, nk; method=method, algorithm=algorithm, normalize=normalize, scale=false, maxiter=maxiter, tol=tol, weightinverse=weightinverse, quiet=quiet, kw...)
	elseif method == :simple
		W, H, objvalue = NMFk.NMFmultiplicative(Xn, nk; quiet=quiet, tol=tol, maxiter=maxiter, kw...)
		objvalue = sum((X - W * H).^2)
		# objvalue = vecnorm(X - W * H) # Frobenius norm is sum((X - W * H).^2)^(1/2) but why bother
	elseif method == :nmf
		W, H = NMF.randinit(Xn, nk)
		if algorithm == :multdiv
			nmf_result = NMF.solve!(NMF.MultUpdate{typeof(X[1,1])}(obj=:mse, maxiter=maxiter, tol=tol), Xn, W, H)
		elseif algorithm == :multmse
			nmf_result = NMF.solve!(NMF.MultUpdate{typeof(X[1,1])}(obj=:div, maxiter=maxiter, tol=tol), Xn, W, H)
		elseif algorithm == :alspgrad
			nmf_result = NMF.solve!(NMF.ALSPGrad{typeof(X[1,1])}(maxiter=maxiter, tol=tol, tolg=tol*100), Xn, W, H)
		end
		!quiet && println("NMF Converged: " * string(nmf_result.converged))
		W = nmf_result.W
		H = nmf_result.H
		objvalue = nmf_result.objvalue
	else
		error("Unknown method: $method")
	end
	if scale
		if transpose
			W = NMFk.descalematrix!(W, Xmax')
			E = X' - W * H
		else
			H = NMFk.descalematrix!(H, Xmax)
			E = X - W * H
		end
		objvalue = sum(E.^2)
	end
	!quiet && println("Objective function = $(objvalue)")
	if mixture == :null
		if clusterweights
			total = sum(W, 1)
			W ./= total
			H .*= total'
		else
			total = sum(H, 2)
			W .*= total'
			H ./= total
		end
	end
	return W, H, objvalue
end

function NMFrun(X::Matrix, nk::Integer; maxiter::Integer=maxiter, normalize::Bool=true)
	W, H = NMF.randinit(X, nk, normalize = true)
	NMF.solve!(NMF.MultUpdate(obj = :mse, maxiter=maxiter), X, W, H)
	if normalize
		total = sum(W, 1)
		W ./= total
		H .*= total'
	end
	return W, H
end

function vecnormnan(X)
	vecnorm(X[.!isnan.(X)])
end

function cornan(x, y)
	isn = isnan.(x) .| isnan.(y)
	cov(x[.!isn], y[.!isn])
end
