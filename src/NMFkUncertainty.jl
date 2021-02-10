"Execute NMFk analysis for a given of number of signals multiple times"
function uncertainty(X::AbstractArray{T,N}, nk::Integer, nreruns::Integer, nNMF::Integer=1; window::Integer=size(X, 1), maxwindow::Integer=window, save::Bool=false, saveall::Bool=false, loadall::Bool=false, resultdir::AbstractString=".", casefilename::AbstractString="", quiet::Bool=false, kw...) where {T, N}
	if casefilename == ""
		casefilename = "nmfk_uncertainty"
	end
	if loadall
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nreruns-$nNMF-all.jld")
		if isfile(filename)
			W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
			return W, H, fitquality, robustness, aic
		else
			@warn("Filename $filename is missing!")
		end
	end
	W = Vector{Array{T, N}}(undef, nreruns)
	H = Vector{Array{T, 2}}(undef, nreruns)
	fitquality = Vector{T}(undef, nreruns)
	robustness = Vector{T}(undef, nreruns)
	aic = Vector{T}(undef, nreruns)
	for i in 1:nreruns
		!quiet && @info("Uncertainty run $(i) out of $(nreruns):")
		casefilenamemod = save == true ? casefilename * "_$i" : ""
		if quiet
			@Suppressor.suppress W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X[1:window,:], nk, nNMF; kw..., save=save, resultdir=resultdir, casefilename=casefilenamemod)
		else
			W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X[1:window,:], nk, nNMF; kw..., save=save, resultdir=resultdir, casefilename=casefilenamemod)
		end
	end
	if !quiet
		if window == size(X,1)
			@info("Uncertainty results:")
		else
			@info("Uncertainty results stage #1:")
		end
		for i in 1:nreruns
			println("Run: $(@Printf.sprintf("%5d", i)) Fit: $(@Printf.sprintf("%12.7g", fitquality[i])) Silhouette: $(@Printf.sprintf("%12.7g", robustness[i])) AIC: $(@Printf.sprintf("%12.7g", aic[i]))")
		end
	end
	if window != size(X,1)
		for i in 1:nreruns
			!quiet && @info("Uncertainty run stage #2 $(i) out of $(nreruns):")
			casefilenamemod = save == true ? casefilename * "_stage2_$i" : ""
			if quiet
				@Suppressor.suppress W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X[1:maxwindow,:], nk; Hinit=convert.(T, H[i]), Hfixed=true, kw..., save=save, resultdir=resultdir, casefilename=casefilenamemod)
			else
				W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X[1:maxwindow,:], nk; Hinit=convert.(T, H[i]), Hfixed=true, kw..., save=save, resultdir=resultdir, casefilename=casefilenamemod)s
			end
		end
		if !quiet
			@info("Uncertainty results stage #2:")
			for i in 1:nreruns
			println("Run: $(@Printf.sprintf("%5d", i)) Fit: $(@Printf.sprintf("%12.7g", fitquality[i])) Silhouette: $(@Printf.sprintf("%12.7g", robustness[i])) AIC: $(@Printf.sprintf("%12.7g", aic[i]))")
			end
		end
	end
	if saveall
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nreruns-$nNMF-all.jld")
		recursivemkdir(filename)
		JLD.save(filename, "W", W, "H", H, "fit", fitquality, "robustness", robustness, "aic", aic)
		@info("Results saved in $filename.")
	end
	return W, H, fitquality, robustness, aic
end