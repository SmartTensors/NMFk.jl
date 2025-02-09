import Distributed
import JLD

function input_checks(X::AbstractArray{T,N}, load::Bool, save::Bool, casefilename::AbstractString, mixture::Symbol, method::Symbol, algorithm::Symbol, clusterWmatrix::Bool) where {T <: Number, N}
	global first_warning = true
	if load && casefilename == ""
		@info("Loading of existing results is requested but \`casefilename\` is not specified; casefilename = \"nmfk\" will be used!")
		casefilename = "nmfk"
	end
	if save && casefilename == ""
		@info("Saving of obtained results is requested but \`casefilename\` is not specified; casefilename = \"nmfk\" will be used!")
		casefilename = "nmfk"
	end
	if mixture != :null
		clusterWmatrix = true
		method = :ipopt
	elseif N > 2
		@error("NMFk analysis can be executed for matrices!")
		@info("For multi-dimensional arrays (tensors), use NMFk.tensorfactorization or NTFk!")
		throw(ArgumentError("NMFk analysis can be executed for matrices!"))
	end
	if N == 2 && size(X, 1) < size(X, 2) && clusterWmatrix == false
		@warn("Processed matrix size has more columns than rows (matrix size=$(size(X)))!")
		@info("In this case, it is recommended to use `clusterWmatrix == true`.")
		@info("It is preferred to cluster the smaller of the matrices!")
	end
	if any(isnan.(X))
		@info("Analyzed matrix has NaN's.")
		if method != :simple && method != :ipopt && method != :nlopt
			@warn("Analyzed matrix has NaN's! NMF method $(method)) cannot be used! Simple multiplicative NMF will be performed!")
			method = :simple
		end
	end
	if method == :nlopt && algorithm == :multdiv
		algorithm = :LD_SLSQP
	end
	if method == :multdiv
		algorithm = method
		method = :nmf
	elseif method == :multmse
		algorithm = method
		method = :nmf
	elseif method == :alspgrad
		algorithm = method
		method = :nmf
	end
	print("$(Base.text_colors[:cyan])$(Base.text_colors[:bold])[ Info: $(Base.text_colors[:normal])")
	if mixture == :mixmatch
		method = :ipopt
		print("MixMatch geochemical analysis using ")
	elseif mixture == :matchwaterdeltas
		print("MixMatchDeltas geochemical analysis using ")
	else
		print("NMFk analysis using ")
	end
	if method == :ipopt
		println("Ipopt ...")
	elseif method == :nlopt
		println("NLopt ...")
	elseif method == :nmf
		if algorithm == :multdiv
			println("Multiplicative updates based on divergence ...")
		elseif algorithm == :multmse
			println("Multiplicative updates based on mean-squared-error ...")
		elseif algorithm == :alspgrad
			println("Alternate Least Square based on Projected Gradient Descent ...")
		end
	elseif method == :sparsity
		println("Sparsity penalty ...")
	elseif method == :simple
		println("Simple multiplicative updates ...")
	end
	return load, save, casefilename, mixture, method, algorithm, clusterWmatrix
end

"Execute NMFk analysis for a range of number of signals"
function execute(X::AbstractArray{T,N}, nkrange::Union{Vector{Int},AbstractRange{Int}}, nNMF::Integer=10; cutoff::Number=0.5, clusterWmatrix::Bool=false,  mixture::Symbol=:null, method::Symbol=:simple, algorithm::Symbol=:multdiv, load::Bool=true, save::Bool=true, casefilename::AbstractString="", kw...) where {T <: Number, N}
	load, save, casefilename, mixture, method, algorithm, clusterWmatrix = input_checks(X, load, save, casefilename, mixture, method, algorithm, clusterWmatrix)
	maxk = maximum(collect(nkrange))
	W = Vector{Array{T, N}}(undef, maxk)
	H = Vector{Matrix{T}}(undef, maxk)
	fitquality = zeros(T, maxk)
	robustness = zeros(T, maxk)
	aic = zeros(T, maxk)
	for nk in nkrange
		W[nk], H[nk], fitquality[nk], robustness[nk], aic[nk] = NMFk.execute(X, nk, nNMF; load=load, save=save, casefilename=casefilename, mixture=mixture, method=method, algorithm=algorithm, clusterWmatrix=clusterWmatrix, check_inputs=false, kw...)
	end
	@info("Results:")
	for nk in nkrange
		println("Signals: $(Printf.@sprintf("%2d", nk)) Fit: $(Printf.@sprintf("%12.7g", fitquality[nk])) Silhouette: $(Printf.@sprintf("%12.7g", robustness[nk])) AIC: $(Printf.@sprintf("%12.7g", aic[nk]))")
	end
	kopt = getk(nkrange, robustness[nkrange], cutoff)
	if isnothing(kopt)
		@warn("No optimal solutions")
	else
		@info("Optimal solution: $kopt signals")
	end
	return W, H, fitquality, robustness, aic, kopt
end

"Execute NMFk analysis for a given number of signals"
function execute(X::AbstractArray{T,N}, nk::Integer, nNMF::Integer=10; clusterWmatrix::Bool=false, mixture::Symbol=:null, method::Symbol=:simple, algorithm::Symbol=:multdiv, resultdir::AbstractString=".", casefilename::AbstractString="", loadonly::Bool=false, load::Bool=true, save::Bool=true, quiet::Bool=false, check_inputs::Bool=true, ordersignals::Bool=true, kw...) where {T <: Number, N}
	if .*(size(X)...) == 0
		error("Input array has a zero dimension! Array size=$(size(X))")
	end
	if loadonly
		load = true
		save = false
		runflag = false
	else
		runflag = true
	end
	if check_inputs
		load, save, casefilename, mixture, method, algorithm, clusterWmatrix = input_checks(X, load, save, casefilename, mixture, method, algorithm, clusterWmatrix)
	end
	print("$(Base.text_colors[:cyan])$(Base.text_colors[:bold])NMFk run with $(nk) signals: $(Base.text_colors[:normal])")
	if load
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF.jld")
		if isfile(filename)
			W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
			if size(W) == (size(X, 1), nk) && size(H) == (nk, size(X, 2))
				print("Results are loaded from $(filename)!")
				save = false
				runflag = false
			else
				print("File $(filename) contains  $(Base.text_colors[:yellow])$(Base.text_colors[:bold])inconsistent results$(Base.text_colors[:normal]); runs will be executed ...")
				println("W matrix - Expected size: $((size(X, 1), nk)) Actual size: $(size(W))")
				println("H matrix - Expected size: $((nk, size(X, 2))) Actual size: $(size(H))")
			end
		else
			print("File $(filename) is $(Base.text_colors[:yellow])$(Base.text_colors[:bold])missing$(Base.text_colors[:normal]); runs will be executed ...")
			if loadonly
				W = Matrix{T}(undef, 0, 0);
				H = Matrix{T}(undef, 0, 0);
				fitquality = Inf;
				robustness = -1;
				aic = -Inf;
			end
		end
	end
	println()
	if haskey(kw, :Wfixed) || haskey(kw, :Hfixed)
		ordersignals = false
	end
	if runflag
		W, H, fitquality, robustness, aic = NMFk.execute_run(X, nk, nNMF; clusterWmatrix=clusterWmatrix, resultdir=resultdir, casefilename=casefilename, mixture=mixture, method=method, algorithm=algorithm, quiet=quiet, kw...)
	end
	if ordersignals
		@info("Ordering signals based on their contribution ...")
		so = signalorder(W, H)
	else
		@warn("Signals are not orered ...")
		so = 1:size(W, 2)
	end
	!quiet && println("Signals: $(Printf.@sprintf("%2d", nk)) Fit: $(Printf.@sprintf("%12.7g", fitquality)) Silhouette: $(Printf.@sprintf("%12.7g", robustness)) AIC: $(Printf.@sprintf("%12.7g", aic)) Signal order: $(so)")
	if save
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF.jld")
		JLD.save(filename, "W", W[:,so], "H", H[so,:], "fit", fitquality, "robustness", robustness, "aic", aic)
		!quiet && @info("Results are saved in $(filename)!")
	end
	return W[:,so], H[so,:], fitquality, robustness, aic
end

"Execute NMFk analysis for a given number of signals in serial or parallel"
function execute_run(X::AbstractArray{T,N}, nk::Int, nNMF::Int; clusterWmatrix::Bool=false, acceptratio::Number=1, acceptfactor::Number=Inf, quiet::Bool=NMFk.global_quiet, veryquiet::Bool=true, best::Bool=true, serial::Bool=false, resultdir::AbstractString=".", casefilename::AbstractString="", loadonly::Bool=false, loadall::Bool=false, saveall::Bool=false, kw...) where {T <: Number, N}
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && @info("Mixmatch geochemical analysis of $nNMF NMF runs assuming $nk signals (sources) ...")
	if loadonly
		saveall = false
		runflag = false
	else
		runflag = true
	end
	if loadall && casefilename != ""
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF-all.jld")
		if isfile(filename)
			@info("All tesults are loaded from $(filename)!")
			WBig, HBig, objvalue = JLD.load(filename, "W", "H", "fit")
			saveall = false
			runflag = false
		else
			@warn("File $(filename) with ALL results is missing; runs will be executed!")
		end
	end
	if runflag
		if Distributed.nprocs() > 1 && !serial
			!quiet && println("Parallel execution of $nNMF NMF runs ...")
			if haskey(kw, :seed)
				kwseed = kw[:seed]
				delete!(kw, :seed)
				r = Distributed.pmap(i->(NMFk.execute_singlerun(X, nk; quiet=true, seed=kwseed+i, kw...)), 1:nNMF)
			else
				r = Distributed.pmap(i->(NMFk.execute_singlerun(X, nk; quiet=true, kw...)), 1:nNMF)
			end
			WBig = Vector{Array{T}}(undef, nNMF)
			HBig = Vector{Matrix{T}}(undef, nNMF)
			for i in 1:nNMF
				WBig[i] = r[i][1]
				HBig[i] = r[i][2]
			end
			objvalue = map(i->convert(T, r[i][3]), 1:nNMF)
		else
			!quiet && println("Serial execution of $nNMF NMF runs ...")
			WBig = Vector{Array{T}}(undef, nNMF)
			HBig = Vector{Matrix{T}}(undef, nNMF)
			objvalue = Vector{T}(undef, nNMF)
			if haskey(kw, :seed)
				kwseed = kw[:seed]
				delete!(kw, :seed)
				for i = 1:nNMF
					!quiet && @info("NMF run #$(i)")
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; quiet=quiet, seed=kwseed+i, kw...)
				end
			else
				for i = 1:nNMF
					!quiet && @info("NMF run #$(i)")
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
	Xe = NMFk.mixmatchcompute(X, Wbest, Hbest)
	!veryquiet && println()
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
		if clusterWmatrix && sum(isnan.(WBig[i])) > 0
			idxnan[idxsort[i]] = false
		end
		if !clusterWmatrix && sum(isnan.(HBig[i])) > 0
			idxnan[idxsort[i]] = false
		end
	end
	if sum(idxnan) < nNMF
		println("NMF solutions removed because they contain NaN's: $(sum(idxnan)) out of $(nNMF) solutions remain")
	end
	idxsol = idxrat .& idxcut .& idxnan
	if sum(idxsol) < nNMF
		@warn("NMF solutions removed based on acceptance criteria: $(sum(idxsol)) out of $(nNMF) solutions remain")
		!veryquiet && println("OF: min $(minimum(objvalue)) max $(maximum(objvalue)) mean $(Statistics.mean(objvalue)) std $(Statistics.std(objvalue)) (ALL)")
	end
	minsilhouette = -1
	if sum(idxnan) > 0
		!veryquiet && println("OF: min $(minimum(objvalue[idxsol])) max $(maximum(objvalue[idxsol])) mean $(Statistics.mean(objvalue[idxsol])) std $(Statistics.std(objvalue[idxsol]))")
		if nk > 1
			clusterassignments, clustercentroids = NMFk.clustersolutions(HBig[idxsort][idxsol], false)
			if !quiet
				@info("Cluster assignments:")
				display(clusterassignments)
				@info("Cluster centroids:")
				display(clustercentroids)
			end
			ci = clusterassignments[:, 1]
			for (i, c) in enumerate(ci)
				if N == 2
					Wbest[:, i] = WBig[bestIdx][:, c]
					Hbest[i, :] = HBig[bestIdx][c, :]
				else
					nti = ntuple(k->(k == 2 ? i : Colon()), N)
					ntc = ntuple(k->(k == 2 ? c : Colon()), N)
					Wbest[nti...] = WBig[bestIdx][ntc...]
					Hbest[i, :] = HBig[bestIdx][c, :]
				end
			end
			Wa, Ha, clustersilhouettes, Wv, Hv = NMFk.finalize(WBig[idxsort][idxsol], HBig[idxsort][idxsol], clusterassignments, false)
			minsilhouette = minimum(clustersilhouettes)
			if !quiet
				@info("Silhouettes for each of the $nk clusters:" )
				display(permutedims(clustersilhouettes))
				println("Mean silhouette = ", Statistics.mean(clustersilhouettes))
				println("Min  silhouette = ", minimum(clustersilhouettes))
			end
		else
			Wv = NaN
			Hv = NaN
			Wa, Ha = NMFk.finalize(WBig[idxsol], HBig[idxsol])
		end
		if saveall && casefilename != ""
			filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF-all.jld")
			JLD.save(filename, "W", WBig, "H", HBig, "Wmean", Wa, "Hmean", Ha, "Wvar", Wv, "Hvar", Hv, "Wbest", Wbest, "Hbest", Hbest, "fit", objvalue, "Cluster Silhouettes", clustersilhouettes, "Cluster assignments", clusterassignments, "Cluster centroids", clustercentroids)
			@info("All results are saved in $(filename)!")
		end
	end
	if best
		Wa = Wbest
		Ha = Hbest
	end
	phi_final = ssqrnan(X .- Xe)
	numobservations = sum(.!isnan.(X))
	numparameters = *(collect(size(Wa))...) + *(collect(size(Ha))...)
	numparameters -= (size(Wa)[1] + size(Wa)[3])
	aic = 2 * numparameters + numobservations * log(phi_final/numobservations)
	E = X .- Xe
	!quiet && println("Objective function = ", phi_final, " Max error = ", maximumnan(E), " Min error = ", minimumnan(E))
	return Wa, Ha, phi_final, minsilhouette, aic
end
function execute_run(X::AbstractMatrix{T}, nk::Int, nNMF::Int; clusterWmatrix::Bool=false, acceptratio::Number=1, acceptfactor::Number=Inf, quiet::Bool=NMFk.global_quiet, veryquiet::Bool=true, best::Bool=true, transpose::Bool=false, serial::Bool=false, deltas::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), ratios::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), mixture::Symbol=:null, resultdir::AbstractString=".", casefilename::AbstractString="", nanaction::Symbol=:zeroed, loadall::Bool=false, saveall::Bool=false, weight=1, kw...) where {T <: Number}
	@assert typeof(weight) <: Number || length(weight) == size(X, 1) || size(weight, 2) == size(X, 2) || size(weight) == size(X)
	quiet = veryquiet ? true : quiet
	modifymatrices = true
	if haskey(kw, :Wfixed) || haskey(kw, :Hfixed)
		modifymatrices = false
	end
	# ipopt=true is equivalent to mixmatch = true && mixtures = false
	!quiet && @info("NMFk analysis of $nNMF NMF runs assuming $nk signals (sources) ...")
	if transpose
		nC, nP = size(X) # number of observed components/transients, number of observation points
	else
		nP, nC = size(X) # number of observation points,  number of observed components/transients
	end
	# nRC = sizeof(deltas) == 0 ? nC : nC + size(deltas, 2)
	runflag = true
	if loadall && casefilename != ""
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF-all.jld")
		if isfile(filename)
			@info("All results are loaded from $(filename)!")
			WBig, HBig, objvalue = JLD.load(filename, "W", "H", "fit")
			saveall = false
			runflag = false
		else
			@warn("File $(filename) with ALL results is missing; runs will be executed!")
		end
	end
	if runflag
		if Distributed.nprocs() > 1 && !serial
			!quiet && println("Parallel execution of $(nNMF) NMF runs ...")
			if haskey(kw, :seed)
				kwseed = kw[:seed]
				delete!(kw, :seed)
				r = Distributed.pmap(i->(NMFk.execute_singlerun(X, nk; modifymatrices=modifymatrices, quiet=true, transpose=transpose, deltas=deltas, ratios=ratios, weight=weight, seed=kwseed+i, kw...)), 1:nNMF)
			else
				r = Distributed.pmap(i->(NMFk.execute_singlerun(X, nk; modifymatrices=modifymatrices, quiet=true, transpose=transpose, deltas=deltas, ratios=ratios, weight=weight, kw...)), 1:nNMF)
			end
			WBig = Vector{Matrix{T}}(undef, nNMF)
			HBig = Vector{Matrix{T}}(undef, nNMF)
			for i in 1:nNMF
				WBig[i] = r[i][1]
				HBig[i] = r[i][2]
			end
			objvalue = map(i->convert(T, r[i][3]), 1:nNMF)
		else
			!quiet && println("Serial execution of $(nNMF) NMF runs ...")
			WBig = Vector{Matrix{T}}(undef, nNMF)
			HBig = Vector{Matrix{T}}(undef, nNMF)
			objvalue = Vector{T}(undef, nNMF)
			if haskey(kw, :seed)
				kwseed = kw[:seed]
				delete!(kw, :seed)
				for i = 1:nNMF
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; modifymatrices=modifymatrices, quiet=quiet, transpose=transpose, deltas=deltas, ratios=ratios, weight=weight, seed=kwseed+i, kw...)
				end
			else
				for i = 1:nNMF
					WBig[i], HBig[i], objvalue[i] = NMFk.execute_singlerun(X, nk; modifymatrices=modifymatrices, quiet=quiet, transpose=transpose, deltas=deltas, ratios=ratios, weight=weight, kw...)
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
	!veryquiet && println()
	if acceptratio < 1
		ccc = convert(Int, (ceil(nNMF * acceptratio)))
		idxrat = vec([trues(ccc); falses(nNMF-ccc)])
		@warn("NMF solutions removed based on an acceptance ratio: $(sum(idxrat)) out of $(nNMF) solutions remain")
	else
		idxrat = trues(nNMF)
	end
	if acceptfactor < Inf
		cutoff = objvalue[bestIdx] * acceptfactor
		idxcut = objvalue[idxsort] .< cutoff
		@warn("NMF solutions removed based on an acceptance factor: $(sum(idxcut)) out of $(nNMF) solutions remain")
	else
		idxcut = trues(nNMF)
	end
	idxnan = trues(nNMF)
	if nanaction == :zeroed
		zerod = 0
		for i in idxsort
			isnw = isnan.(WBig[i])
			WBig[i][isnw] .= 0
			isnh = isnan.(HBig[i])
			HBig[i][isnh] .= 0
			if sum(isnw) > 0 || sum(isnh) > 0
				zerod += 1
			end
		end
		if zerod > 0
			@warn("NMF solutions contain NaN's: $(zerod) out of $(nNMF) solutions! NaN's have been converted to zeros!")
		end
	elseif nanaction == :removed
		for i in idxsort
			isnw = isnan.(WBig[i])
			if sum(isnw) > 0
				idxnan[i] = false
			end
			isnh = isnan.(HBig[i])
			if sum(isnh) > 0
				idxnan[i] = false
			end
		end
		if sum(idxnan) < nNMF
			@warn("NMF solutions removed because they contain NaN's: $(sum(idxnan)) out of $(nNMF) solutions remain")
		end
	end
	idxsol = idxrat .& idxcut .& idxnan
	if sum(idxsol) < nNMF
		println("NMF solutions removed based on various criteria: $(sum(idxsol)) out of $(nNMF) solutions remain")
		println("OF: min $(minimum(objvalue)) max $(maximum(objvalue)) mean $(Statistics.mean(objvalue)) std $(Statistics.std(objvalue)) (ALL)")
	end
	!veryquiet && println("OF: min $(minimum(objvalue[idxsol])) max $(maximum(objvalue[idxsol])) mean $(Statistics.mean(objvalue[idxsol])) std $(Statistics.std(objvalue[idxsol]))")
	for i in 1:nNMF
		of = ssqrnan((X - WBig[i] * HBig[i]) .* weight)
		if abs(of - objvalue[i]) / of > 1e-4
			@warn("OF $i is very different: $(of) vs $(objvalue[i])!")
		end
	end
	Xe = Wbest * Hbest
	fn = normnan(X .- Xe)
	if !veryquiet
		println("Worst correlation by columns: $(minimumnan(map(i->cornan(X[i, :], Xe[i, :]), 1:size(X, 1))))")
		println("Worst correlation by rows: $(minimumnan(map(i->cornan(X[:, i], Xe[:, i]), 1:size(X, 2))))")
		println("Worst covariance by columns: $(minimumnan(map(i->covnan(X[i, :], Xe[i, :]), 1:size(X, 1))))")
		println("Worst covariance by rows: $(minimumnan(map(i->covnan(X[:, i], Xe[:, i]), 1:size(X, 2))))")
		println("Worst norm by columns: $(maximumnan(map(i->(normnan(X[i, :] - Xe[i, :]) / fn), 1:size(X, 1))))")
		println("Worst norm by rows: $(maximumnan(map(i->(normnan(X[:, i] - Xe[:, i]) / fn), 1:size(X, 2))))")
	end
	minsilhouette = 1
	if nk > 1
		if clusterWmatrix
			clusterassignments, clustercentroids = NMFk.clustersolutions(WBig[idxsort][idxsol], clusterWmatrix)
		else
			clusterassignments, clustercentroids = NMFk.clustersolutions(HBig[idxsort][idxsol], clusterWmatrix)
		end
		if !quiet
			@info("Cluster assignments:")
			display(clusterassignments)
			@info("Cluster centroids:")
			display(clustercentroids)
		end
		ci = clusterassignments[:, 1]
		for (i, c) in enumerate(ci)
			Wbest[:, i] = WBig[bestIdx][:, c]
			Hbest[i, :] = HBig[bestIdx][c, :]
		end
		Xe = Wbest * Hbest
		Wa, Ha, clustersilhouettes, Wv, Hv = NMFk.finalize(WBig[idxsort][idxsol], HBig[idxsort][idxsol], clusterassignments, clusterWmatrix)
		minsilhouette = minimum(clustersilhouettes)
		if !quiet
			@info("Silhouettes for each of the $nk clusters:" )
			display(permutedims(clustersilhouettes))
			println("Mean silhouette = ", Statistics.mean(clustersilhouettes))
			println("Min  silhouette = ", minimum(clustersilhouettes))
		end
	else
		Wv = NaN
		Hv = NaN
		Wa, Ha = NMFk.finalize(WBig[idxsol], HBig[idxsol])
	end
	if saveall && casefilename != ""
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF-all.jld")
		JLD.save(filename, "W", WBig, "H", HBig, "Wmean", Wa, "Hmean", Ha, "Wvar", Wv, "Hvar", Hv, "Wbest", Wbest, "Hbest", Hbest, "fit", objvalue, "Cluster Silhouettes", clustersilhouettes, "Cluster assignments", clusterassignments, "Cluster centroids", clustercentroids)
		@info("All results are saved in $(filename)!")
	end
	if best
		Wa = Wbest
		Ha = Hbest
	end
	Xe = Wbest * Hbest
	if sizeof(deltas) == 0
		if transpose
			E = permutedims(X) - Wa * Ha
		else
			E = X - Wa * Ha
		end
		E[isnan.(E)] .= 0
		phi_final = sum(E.^2)
	else
		Ha_conc = Ha[:,1:nC]
		Ha_deltas = Ha[:,nC+1:end]
		estdeltas = NMFk.computedeltas(Wa, Ha_conc, Ha_deltas, deltaindices)
		if transpose
			E = permutedims(X) - Wa * Ha_conc
		else
			E = X - Wa * Ha_conc
		end
		E[isnan.(E)] .= 0
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
	numobservations = sum(.!isnan.(X))
	numparameters = *(collect(size(Wa))...) + *(collect(size(Ha))...)
	if mixture != :null
		numparameters -= size(Wa)[1]
	end
	# numparameters = numbuckets # this is wrong
	# dof = numobservations - numparameters # this is correct, but we cannot use because we may get negative DoF
	# dof = maximum(nkrange) - numparameters + 1 # this is a hack to make the dof positive.
	# dof = dof < 0 ? 0 : dof
	# sml = dof + numobservations * (log(fitquality[numbuckets]/dof) / 2 + 1.837877)
	# aic[numbuckets] = sml + 2 * numparameters
	aic = 2 * numparameters + numobservations * log(phi_final/numobservations)
	!quiet && println("Objective function = ", phi_final, " Max error = ", maximumnan(E), " Min error = ", minimumnan(E) )
	return Wa, Ha, phi_final, minsilhouette, aic
end

"Execute single NMF run"
function execute_singlerun(x...; kw...)
	if restart
		@info("NMF runs with efficient restarts ...")
		return execute_singlerun_r3(x...; kw...)
	else
		return execute_singlerun_compute(x...; kw...)
	end
end

"Execute single NMF run without restart"
function execute_singlerun_compute(X::AbstractArray, nk::Int; kw...)
	NMFk.mixmatchdata(X, nk; kw...)
end

"Execute single NMF run without restart"
function execute_singlerun_compute(X::AbstractMatrix{T}, nk::Int; quiet::Bool=NMFk.global_quiet, ratios::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), ratioindices::Union{AbstractVector{Int},AbstractMatrix{Int}}=Matrix{Int}(undef, 0, 0), deltas::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), deltaindices::AbstractVector{Int}=Vector{Int}(undef, 0), transpose::Bool=false, scale::Bool=false, modifymatrices::Bool=true, maxiter::Int=10000, tol::Float64=1e-19, ratiosweight::T=convert(T, 1), weightinverse::Bool=false, mixture::Symbol=:null, method::Symbol=:simple, algorithm::Symbol=:multdiv, clusterWmatrix::Bool=false, bootstrap::Bool=false, weight=1, kw...) where {T <: Number}
	if scale
		if transpose
			Xn, Xmax = NMFk.scalematrix_row!(permutedims(X))
		else
			Xn, Xmax = NMFk.scalematrix_row!(X)
		end
	else
		if transpose
			Xn = permutedims(X)
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
				W, H, objvalue = NMFk.mixmatchdata(Xn, nk; method=method, algorithm=algorithm, ratios=ratios, ratioindices=ratioindices, maxiter=maxiter, weightinverse=weightinverse, ratiosweight=ratiosweight, quiet=quiet, tol=tol, kw...)
			else
				W, Hconc, Hdeltas, objvalue = NMFk.mixmatchdeltas(Xn, deltas, deltaindices, nk; method=method, algorithm=algorithm, maxiter=maxiter, weightinverse=weightinverse, quiet=quiet, tol=tol, kw...)
				H = [Hconc Hdeltas]
			end
		elseif mixture == :matchwaterdeltas
			W, H, objvalue = NMFk.mixmatchwaterdeltas(Xn, nk; method=method, algorithm=algorithm, maxiter=maxiter, kw...)
		end
	elseif method == :sparsity
		W, H, objvalue = NMFk.NMFsparsity(Xn, nk; maxiter=maxiter, tol=tol, quiet=quiet, kw...)
	elseif method == :ipopt || method == :nlopt
		W, H, objvalue = NMFk.jump(Xn, nk; method=method, algorithm=algorithm, maxiter=maxiter, tol=tol, weightinverse=weightinverse, quiet=quiet, kw...)
	elseif method == :simple
		W, H, objvalue = NMFk.NMFmultiplicative(Xn, nk; quiet=quiet, tol=tol, maxiter=maxiter, weight=weight, kw...)
	elseif method == :nmf
		W, H = NMF.randinit(Xn, nk)
		if algorithm == :multdiv
			nmf_result = NMF.solve!(NMF.MultUpdate{eltype(X)}(obj=:mse, maxiter=maxiter, tol=tol), Xn, W, H)
		elseif algorithm == :multmse
			nmf_result = NMF.solve!(NMF.MultUpdate{eltype(X)}(obj=:div, maxiter=maxiter, tol=tol), Xn, W, H)
		elseif algorithm == :alspgrad
			nmf_result = NMF.solve!(NMF.ALSPGrad{eltype(X)}(maxiter=maxiter, tol=tol, tolg=tol*100), Xn, W, H)
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
			X = NMFk.descalematrix!(Xn, Xmax)
			W = NMFk.descalematrix!(W, permutedims(Xmax))
			E = permutedims(X) - W * H
		else
			X = NMFk.descalematrix!(Xn, Xmax)
			H = NMFk.descalematrix!(H, Xmax)
			E = X - W * H
		end
		objvalue = sum(E.^2)
	else
		E = X - W * H
	end
	!quiet && println("Objective function = $(objvalue)")
	if mixture == :null && modifymatrices
		if clusterWmatrix
			total = sum(W; dims=1)
			W ./= total
			H .*= permutedims(total)
		else
			total = sum(H; dims=2)
			W .*= permutedims(total)
			H ./= total
		end
	end
	return W, H, objvalue
end

function NMFrun(X::AbstractMatrix, nk::Integer; maxiter::Integer=maxiter, normalize::Bool=true)
	W, H = NMF.randinit(X, nk; normalize=normalize)
	NMF.solve!(NMF.MultUpdate{eltype(X)}(obj=:mse, maxiter=maxiter), X, W, H)
	if normalize
		total = sum(W, 1)
		W ./= total
		H .*= permutedims(total)
	end
	return W, H
end