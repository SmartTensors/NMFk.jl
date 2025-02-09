import JLD
import Mads

function load(nkrange::AbstractRange{Int}, nNMF::Integer=10; cutoff::Number=0.5, quiet::Bool=false, strict::Bool=true, kw...)
	maxsignals = maximum(collect(nkrange))
	aicl = NaN
	igood = 0
	local Wl, Hl, fitqualityl, robustnessl, aicl
	while isnan(aicl) && igood < length(nkrange)
		igood += 1
		Wl, Hl, fitqualityl, robustnessl, aicl = NMFk.load(nkrange[igood], nNMF; quiet=quiet, kw...)
	end
	dim = ndims(Wl)
	type = eltype(Wl)
	W = Vector{Array{type, dim}}(undef, maxsignals)
	H = Vector{Matrix{type}}(undef, maxsignals)
	fitquality = Vector{type}(undef, maxsignals)
	robustness = Vector{type}(undef, maxsignals)
	aic = Vector{type}(undef, maxsignals)
	for k = 1:nkrange[igood]
		W[k], H[k], fitquality[k], robustness[k], aic[k] = Array{type, dim}(undef, [0 for i=1:dim]...), Matrix{type}(undef, 0, 0), NaN, NaN, NaN
	end
	k = nkrange[igood]
	W[k], H[k], fitquality[k], robustness[k], aic[k] = Wl, Hl, fitqualityl, robustnessl, aicl
	for k in nkrange[igood+1:end]
		W[k], H[k], fitquality[k], robustness[k], aic[k] = NMFk.load(k, nNMF; type=type, dim=dim, quiet=quiet, kw...)
	end
	kopt = getk(nkrange, robustness[nkrange], cutoff; strict=strict)
	if !quiet
		if !isnothing(kopt)
			@info("Optimal solution: $kopt signals")
		else
			@info("No optimal solution: $kopt signals")
		end
	end
	return W, H, fitquality, robustness, aic, kopt
end
function load(nk::Integer, nNMF::Integer=10; type::DataType=Float64, dim::Integer=2, resultdir::AbstractString=".", casefilename::AbstractString="nmfk", filename::AbstractString="", quiet::Bool=false, ordersignals::Bool=true)
	if casefilename != "" && filename == ""
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF.jld")
	end
	if isfile(filename)
		W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
		if ordersignals
			@info("Ordering signals ...")
			so = signalorder(W, H)
		else
			@warn("Signals are not orered ...")
			so = 1:size(W, 2)
		end
		!quiet && println("Signals: $(Printf.@sprintf("%2d", nk)) Fit: $(Printf.@sprintf("%12.7g", fitquality)) Silhouette: $(Printf.@sprintf("%12.7g", robustness)) AIC: $(Printf.@sprintf("%12.7g", aic)) Signal order: $(so)")
		return W[:,so], H[so,:], fitquality, robustness, aic
	else
		!quiet && @warn("File named $(filename) is missing!")
		return Array{type, dim}(undef, [0 for i=1:dim]...), Matrix{type}(undef, 0, 0), NaN, NaN, NaN
	end
end

@doc """
Load NMFk analysis results
""" load

function save(t::Tuple, o...; kw...)
	save(t..., o...; kw...)
end
function save(W, H, fitquality, robustness, aic, nkrange::AbstractRange{Int}=1:length(W), nNMF::Integer=10; kw...)
	for nk in nkrange
		if isassigned(W, nk)
			save(W[nk], H[nk], fitquality[nk], robustness[nk], aic[nk], nk, nNMF; kw...)
		end
	end
end
function save(W, H, fitquality, robustness, aic, nk::Integer, nNMF::Integer=10; resultdir=".", casefilename::AbstractString="nmfk", filename::AbstractString="")
	if casefilename != "" && filename == ""
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF.jld")
	else
		recursivemkdir(filename)
	end
	if !isfile(filename)
		@info("Results saved in $(filename) ...")
		JLD.save(filename, "W", W, "H", H, "fit", fitquality, "robustness", robustness, "aic", aic)
	else
		@warn("File named $(filename) already exists!")
	end
end
@doc """
Save NMFk analysis results
""" save

recursivemkdir = Mads.recursivemkdir

function joinpathcheck(path::AbstractString, paths::AbstractString...)
	if path == "." && paths[1] == '/'
		filenamelong = joinpath(paths...)
	else
		filenamelong = joinpath(path, paths...)
	end
	recursivemkdir(filenamelong)
	return filenamelong
end