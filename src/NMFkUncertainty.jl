"Execute NMFk analysis for a given of number of signals multiple times"
function uncertainty(X::AbstractArray{T,N}, nk::Integer, nreruns::Integer, nNMF::Integer=10; save::Bool=false, casefilename::AbstractString="", kw...) where {T, N}
	W = Vector{Array{T, N}}(undef, nreruns)
	H = Vector{Array{T, 2}}(undef, nreruns)
	fitquality = Vector{T}(undef, nreruns)
	robustness = Vector{T}(undef, nreruns)
	aic = Vector{T}(undef, nreruns)
	for i in 1:nreruns
		@info("Rerun $(i) out of $(nreruns):")
		if save == true
			if casefilename == ""
				casefilenamemod = "nmfk_uncertinty_$i"
			else
				casefilenamemod = casefilename * "_$i"
			end
		else
			casefilenamemod = ""
		end
		W[i], H[i], fitquality[i], robustness[i], aic[i] = NMFk.execute(X, nk, nNMF; kw..., save=save, casefilename=casefilenamemod)
	end
	@info("Results")
	for i in 1:nreruns
		println("Signals: $(@sprintf("%2d", i)) Fit: $(@sprintf("%12.7g", fitquality[i])) Silhouette: $(@sprintf("%12.7g", robustness[i])) AIC: $(@sprintf("%12.7g", aic[i]))")
	end
	return W, H, fitquality, robustness, aic
end