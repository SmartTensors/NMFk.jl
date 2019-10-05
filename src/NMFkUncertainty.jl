"Execute NMFk analysis for a given of number of signals multiple times"
function uncertainty(X::AbstractArray{T,N}, nk::Integer, nreruns::Integer, nNMF::Integer=10; window::Integer=size(X, 1), maxwindow::Integer=window, save::Bool=false, casefilename::AbstractString="", kw...) where {T, N}
	W = Vector{Array{T, N}}(undef, nreruns)
	H = Vector{Array{T, 2}}(undef, nreruns)
	fitquality = Vector{T}(undef, nreruns)
	robustness = Vector{T}(undef, nreruns)
	aic = Vector{T}(undef, nreruns)
	for i in 1:nreruns
		@info("Rerun $(i) out of $(nreruns):")
		if save == true
			if casefilename == ""
				casefilenamemod = "nmfk_uncertainty_$i"
			else
				casefilenamemod = casefilename * "_$i"
			end
		else
			casefilenamemod = ""
		end
		W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X[1:window,:], nk, nNMF; kw..., save=save, casefilename=casefilenamemod)
	end
	if window == size(X,1)
		@info("Results")
	else
		@info("Results stage #1")
	end
	for i in 1:nreruns
		println("Rerun: $(@sprintf("%5d", i)) Fit: $(@sprintf("%12.7g", fitquality[i])) Silhouette: $(@sprintf("%12.7g", robustness[i])) AIC: $(@sprintf("%12.7g", aic[i]))")
	end
	if window != size(X,1)
		for i in 1:nreruns
			@info("Rerun stage #2 $(i) out of $(nreruns):")
			if save == true
				if casefilename == ""
					casefilenamemod = "nmfk_uncertainty2_$i"
				else
					casefilenamemod = casefilename * "_stage2_$i"
				end
			else
				casefilenamemod = ""
			end
			W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X[1:maxwindow,:], nk; Hinit=convert.(T, H[i]), Hfixed=true, kw..., save=save, casefilename=casefilenamemod)
		end
		@info("Results stage #2")
		for i in 1:nreruns
			println("Rerun: $(@sprintf("%5d", i)) Fit: $(@sprintf("%12.7g", fitquality[i])) Silhouette: $(@sprintf("%12.7g", robustness[i])) AIC: $(@sprintf("%12.7g", aic[i]))")
		end
	end
	return W, H, fitquality, robustness, aic
end