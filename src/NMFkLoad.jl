function load(nkrange::AbstractRange{Int}, nNMF::Integer=10; kw...)
	maxsources = maximum(collect(nkrange))
	W = Array{Array{Float64, 2}}(undef, maxsources)
	H = Array{Array{Float64, 2}}(undef, maxsources)
	fitquality = Array{Float64}(undef, maxsources)
	robustness = Array{Float64}(undef, maxsources)
	aic = Array{Float64}(undef, maxsources)
	for numsources in nkrange
		W[numsources], H[numsources], fitquality[numsources], robustness[numsources], aic[numsources] = NMFk.load(numsources, nNMF; kw...)
	end
	return W, H, fitquality, robustness, aic
end
function load(nk::Integer, nNMF::Integer=10; resultdir=".", casefilename::AbstractString="nmfk", filename::AbstractString="")
	if casefilename != "" && filename == ""
		filename = joinpath(resultdir, "$casefilename-$nk-$nNMF.jld")
	end
	if isfile(filename)
		W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
		println("Signals: $(@sprintf("%2d", nk)) Fit: $(@sprintf("%12.7g", fitquality)) Silhouette: $(@sprintf("%12.7g", robustness)) AIC: $(@sprintf("%12.7g", aic))")
		return W, H, fitquality, robustness, aic
	else
		@warn("File named $filename is missing!")
		return Array{Float64,2}(undef, 0, 0), Array{Float64,2}(undef, 0, 0), NaN, NaN, NaN
	end
end

@doc """
Load NMFk analysis for a given number of signals
""" load
