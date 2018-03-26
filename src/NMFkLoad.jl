"Execute NMFk analysis for a range of number of sources"
function load(range::Range{Int}, nNMF::Integer=10; kw...)
	maxsources = maximum(collect(range))
	W = Array{Array{Float64, 2}}(maxsources)
	H = Array{Array{Float64, 2}}(maxsources)
	fitquality = Array{Float64}(maxsources)
	robustness = Array{Float64}(maxsources)
	aic = Array{Float64}(maxsources)
	for numsources in range
		W[numsources], H[numsources], fitquality[numsources], robustness[numsources], aic[numsources] = NMFk.load(numsources, nNMF; kw...)
	end
	return W, H, fitquality, robustness, aic
end

"Execute NMFk analysis for a given number of sources"
function load(nk::Integer, nNMF::Integer=10; casefilename::AbstractString="", kw...)
	if casefilename != ""
		filename = "$casefilename-$nk-$nNMF.jld"
		if isfile(filename)
			W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
			println("Sources: $(@sprintf("%2d", nk)) Fit: $(@sprintf("%12.7g", fitquality)) Silhouette: $(@sprintf("%12.7g", robustness)) AIC: $(@sprintf("%12.7g", aic))")
			return W, H, fitquality, robustness, aic
		else
			return Array{Float64,2}(), Array{Float64,2}(), NaN, NaN, NaN
		end
	end
end