import JLD

function load(nkrange::AbstractRange{Int}, nNMF::Integer=10; cutoff::Number=0.5, kw...)
	maxsignals = maximum(collect(nkrange))
	aicl = NaN
	i = 0
	local Wl, Hl, fitqualityl, robustnessl, aicl
	while isnan(aicl) && i < length(nkrange)
		i += 1
		Wl, Hl, fitqualityl, robustnessl, aicl = NMFk.load(nkrange[i], nNMF; kw...)
	end
	dim = ndims(Wl)
	type = eltype(Wl)
	W = Array{Array{type, dim}}(undef, maxsignals)
	H = Array{Array{type, 2}}(undef, maxsignals)
	fitquality = Array{type}(undef, maxsignals)
	robustness = Array{type}(undef, maxsignals)
	aic = Array{type}(undef, maxsignals)
	k = nkrange[i]
	W[k], H[k], fitquality[k], robustness[k], aic[k] = Wl, Hl, fitqualityl, robustnessl, aicl
	for k = 1:i
		W[k], H[k], fitquality[k], robustness[k], aic[k] = Array{type, dim}(undef, [0 for i=1:dim]...), Array{type, 2}(undef, 0, 0), NaN, NaN, NaN
	end
	k = nkrange[i]
	W[k], H[k], fitquality[k], robustness[k], aic[k] = Wl, Hl, fitqualityl, robustnessl, aicl
	for k in nkrange[i+1:end]
		W[k], H[k], fitquality[k], robustness[k], aic[k] = NMFk.load(k, nNMF; type=type, dim=dim, kw...)
	end
	kopt = getk(nkrange, robustness[nkrange], cutoff)
	i < length(nkrange) && @info("Optimal solution: $kopt signals")
	return W, H, fitquality, robustness, aic, kopt
end
function load(nk::Integer, nNMF::Integer=10; type::DataType=Float64, dim::Integer=2, resultdir::AbstractString=".", casefilename::AbstractString="nmfk", filename::AbstractString="", quiet::Bool=false)
	if casefilename != "" && filename == ""
		filename = joinpathcheck(resultdir, "$casefilename-$nk-$nNMF.jld")
	end
	if isfile(filename)
		W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
		so = signalorder(W, H)
		!quiet && println("Signals: $(@Printf.sprintf("%2d", nk)) Fit: $(@Printf.sprintf("%12.7g", fitquality)) Silhouette: $(@Printf.sprintf("%12.7g", robustness)) AIC: $(@Printf.sprintf("%12.7g", aic)) Signal order: $(so)")
		return W[:,so], H[so,:], fitquality, robustness, aic
	else
		@warn("File named $filename is missing!")
		return Array{type, dim}(undef, [0 for i=1:dim]...), Array{type, 2}(undef, 0, 0), NaN, NaN, NaN
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
	end
	recursivemkdir(filename)
	if !isfile(filename)
		@info("Results saved in $filename ...")
		JLD.save(filename, "W", W, "H", H, "fit", fitquality, "robustness", robustness, "aic", aic)
	else
		@warn("File named $filename already exists!")
	end
end

@doc """
Save NMFk analysis results
""" save

"""
Create directories recursively (if does not already exist)

$(DocumentFunction.documentfunction(recursivemkdir;
argtext=Dict("dirname"=>"directory")))
"""
function recursivemkdir(s::AbstractString; filename=true)
	if filename
		if isfile(s)
			return
		end
	else
		if isdir(s)
			return
		end
	end
	d = Vector{String}(undef, 0)
	sc = deepcopy(s)
	if !filename && sc!= ""
		push!(d, sc)
	end
	while true
		sd = splitdir(sc)
		sc = sd[1]
		if sc == "" || sc == "/"
			break;
		end
		push!(d, sc)
	end
	for i = length(d):-1:1
		sc = d[i]
		if isfile(sc)
			@warn("File $(sc) exists!")
			return
		elseif !isdir(sc)
			mkdir(sc)
			@info("Make dir $(sc)")
		end
	end
end

function joinpathcheck(path::AbstractString, paths::AbstractString...)
	if path == "." && paths[1] == '/'
		filenamelong = joinpath(paths...)
	else
		filenamelong = joinpath(path, paths...)
	end
	return filenamelong
end