function load(nkrange::AbstractRange{Int}, nNMF::Integer=10, dim::Integer=2, t::DataType=Float64; kw...)
	maxsources = maximum(collect(nkrange))
	W = Array{Array{t, dim}}(undef, maxsources)
	H = Array{Array{t, 2}}(undef, maxsources)
	fitquality = Array{t}(undef, maxsources)
	robustness = Array{t}(undef, maxsources)
	aic = Array{t}(undef, maxsources)
	for numsources in nkrange
		W[numsources], H[numsources], fitquality[numsources], robustness[numsources], aic[numsources] = NMFk.load(numsources, nNMF, t; dim=dim, kw...)
	end
	kopt = getk(nkrange, robustness[nkrange])
	@info("Optimal solution: $kopt features")
	return W, H, fitquality, robustness, aic, kopt
end
function load(nk::Integer, nNMF::Integer=10, t::DataType=Float64; dim::Integer=2, resultdir::AbstractString=".", casefilename::AbstractString="nmfk", filename::AbstractString="")
	if casefilename != "" && filename == ""
		filename = joinpath(resultdir, "$casefilename-$nk-$nNMF.jld")
	end
	if isfile(filename)
		W, H, fitquality, robustness, aic = JLD.load(filename, "W", "H", "fit", "robustness", "aic")
		println("Signals: $(@Printf.sprintf("%2d", nk)) Fit: $(@Printf.sprintf("%12.7g", fitquality)) Silhouette: $(@Printf.sprintf("%12.7g", robustness)) AIC: $(@Printf.sprintf("%12.7g", aic))")
		return W, H, fitquality, robustness, aic
	else
		@warn("File named $filename is missing!")
		return Array{t, dim}(undef, [0 for i=1:dim]...), Array{t, 2}(undef, 0, 0), NaN, NaN, NaN
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
		filename = joinpath(resultdir, "$casefilename-$nk-$nNMF.jld")
	end
	if !isdir(resultdir)
		recursivemkdir(resultdir; filename=false)
	end
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
function recursivemkdir(s::String; filename=true)
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
		if sc == ""
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
