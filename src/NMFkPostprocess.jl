import DelimitedFiles
import PlotlyJS
import Mads
import Measures
import Clustering # needed for reconstuction of loaded JLD2 files

function getk(nkrange::Union{AbstractUnitRange{T1},AbstractVector{T1}}, robustness::AbstractVector{T2}, cutoff::Number=0.5; strict::Bool=true) where {T1 <: Integer, T2 <: Number}
	if length(nkrange) != length(robustness)
		robustness = robustness[nkrange]
	end
	if all(isnan.(robustness))
		return 0
	end
	if length(nkrange) == 1
		if strict
			if last(robustness) > cutoff
				k = nkrange[end]
			else
				k = nothing
			end
		else
			k = nkrange[end]
		end
	else
		kn = findlast(i->i > cutoff, robustness)
		if isnothing(kn)
			if strict
				k = nothing
			else
				inan = isnan.(robustness)
				robustness[inan] .= -Inf
				kn = findmax(robustness)[2]
				robustness[inan] .= NaN
				k = nkrange[kn]
			end
		else
			k = nkrange[kn]
		end
	end
	return k
end

function getks(nkrange::Union{AbstractUnitRange{T1},AbstractVector{T1}}, robustness::AbstractVector{T2}, cutoff::Number=0.5; ks::Union{T3, AbstractVector{T3}}=Int64[], strict::Bool=true) where {T1 <: Integer, T2 <: Number, T3 <: Integer}
	@assert length(ks) == 0 || maximum(ks) <= maximum(nkrange)
	if length(nkrange) != length(robustness)
		robustness = robustness[nkrange]
	end
	if all(isnan.(robustness))
		return []
	end
	if length(nkrange) == 1
		if strict
			if robustness[end] > cutoff
				k = [nkrange[end]]
			else
				k = []
			end
		else
			k = [nkrange[end]]
		end
		return k
	else
		kn = findall(i->i > cutoff, robustness)
		if length(kn) == 0
			inan = isnan.(robustness)
			robustness[inan] .= -Inf
			kn = last(findmax(robustness))
			robustness[inan] .= NaN
		end
		if length(nkrange) == length(robustness)
			k = nkrange[kn]
		else
			k = kn
		end
		if !(typeof(k) <: AbstractVector)
			k = [k]
		end
		return unique(sort([k; ks]))
	end
end

function getks(nkrange::Union{AbstractUnitRange{T1},AbstractVector{T1}}, F::AbstractVector{T2}, map=Colon(), cutoff::Number=0.25; ks::Union{T3, AbstractVector{T3}}=Int64[], strict::Bool=true) where {T1 <: Integer, T2 <: AbstractArray, T3 <: Integer}
	@assert length(nkrange) == length(F)
	@assert length(ks) == 0 || maximum(ks) <= maximum(nkrange)
	if all(isnan.(robustness))
		return []
	end
	if length(nkrange) == 1
		if strict
			if robustness[end] > cutoff
				k = [nkrange[end]]
			else
				k = []
			end
		else
			k = [nkrange[end]]
		end
		return k
	else
		kn = Vector{Int64}(undef, 0)
		for (i, k) in enumerate(nkrange)
			if size(F[i], 1) == k
				M = F[i] ./ maximum(F[i]; dims=2)
				any(M[:,map] .> cutoff) && push!(kn, k)
			elseif size(F[i], 2) == k
				M = F[i] ./ maximum(F[i]; dims=1)
				any(M[map,:] .> cutoff) && push!(kn, k)
			end
		end
		return unique(sort([k; ks]))
	end
end

function signalrescale!(W::AbstractMatrix, H::AbstractMatrix; Wnormalize::Bool=true, check::Bool=true)
	if check
		X = W * H
	end
	if Wnormalize
		wm = maximum(W; dims=1)
		W ./= wm
		H .*= permutedims(wm)
		wh = maximum(H)
		H ./= wh
		W .*= wh
	else
		hm = maximum(H; dims=2)
		H ./= hm
		W .*= permutedims(hm)
		wm = maximum(W)
		W ./= wm
		H .*= wm
	end
	check && @assert(maximum(abs.((X - W * H))) < 1.)
end

function signalorder(krange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector)
	signal_order = Vector{Vector{Int64}}(undef, maximum(krange))
	for k = 1:maximum(krange)
		signal_order[k] = Vector{Int64}(undef, 0)
	end
	for k in krange
		@info("Number of signals: $k")
		signal_order[k] = signalorder(W[k], H[k])
	end
	return signal_order
end

function signalorder(W::AbstractMatrix, H::AbstractMatrix; quiet::Bool=true)
	k = size(W, 2)
	@assert k == size(H, 1)
	signal_sum = Vector{eltype(W)}(undef, k)
	for i = 1:k
		signal_sum[i] = sum(W[:,i:i] * H[i:i,:])
	end
	signal_order = sortperm(signal_sum; rev=true)
	!quiet && println("Signal importance (high->low): $signal_order")
	return signal_order
end

function signalorderassignments(X::AbstractArray, dim::Integer=1)
	v = Vector{Int64}(undef, size(X, dim))
	for i in axes(X, dim)
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(X))
		v[i] = findmax(X[nt...])[2]
	end
	sortperm(v), v
end

function signalorderassignments(W::AbstractMatrix, H::AbstractMatrix; resultdir::AbstractString=".", loadassignements::Bool=true, Wclusterlabelcasefilename::AbstractString="Wmatrix", Hclusterlabelcasefilename::AbstractString="Hmatrix", repeats::Integer=1000, Wrepeats::Integer=repeats, Hrepeats::Integer=repeats)
	k = size(H, 1)
	Hclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(H, k, Wrepeats; resuldir=resultdir, casefilename=Hclusterlabelcasefilename, load=loadassignements, save=true, compute_silhouettes_flag=false).assignments)
	Hcs = sortperm(Hclusterlabels)
	clusterlabels = sort(unique(Hclusterlabels))
	Hsignalmap = NMFk.signalassignments(H, Hclusterlabels; clusterlabels=clusterlabels, dims=2)
	Hclustermap = Vector{Char}(undef, k)
	Hclustermap .= ' '
	Hsignals = Vector{String}(undef, length(Hclusterlabels))
	for (j, i) in enumerate(clusterlabels)
		Hclustermap[Hsignalmap[j]] = i
		Hsignals[Hclusterlabels .== i] .= "S$(Hsignalmap[j])"
	end
	Wclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(permutedims(W), k, repeats; resultdir=resultdir, casefilename=Wclusterlabelcasefilename, load=loadassignements, save=true, compute_silhouettes_flag=false).assignments)
	@assert clusterlabels == sort(unique(Wclusterlabels))
	Wsignalmap = NMFk.signalassignments(W[:,Hsignalmap], Wclusterlabels; clusterlabels=clusterlabels, dims=1)
	Wclusterlabelsnew = Vector{eltype(Wclusterlabels)}(undef, length(Wclusterlabels))
	Wclusterlabelsnew .= ' '
	Wsignals = Vector{String}(undef, length(Wclusterlabels))
	for (j, i) in enumerate(clusterlabels)
		iclustermap = Wsignalmap[j]
		Wclusterlabelsnew[Wclusterlabels .== i] .= clusterlabels[iclustermap]
		Wsignals[Wclusterlabels .== i] .= "S$(Wsignalmap[j])"
	end
	return Wclusterlabelsnew, Wsignals, Hclusterlabels, Hsignals
end

function signal_statistics(nkrange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector, dim::Integer=2; figuredir::AbstractString=".", casefilename::AbstractString="", plotformat::AbstractString="png", names::AbstractVector, print::Bool=true, func::Function=Statistics.var, map::Function=i->i, kw...)
	for k in nkrange
		Xe = map(W[k] * H[k])
		print && (@info("Reconstructed data:"); display([names Xe']))
		isignalmap = signalorder(W[k], H[k])
		stata = Vector{eltype(W[k])}(undef, size(Xe, dim))
		for i in axes(Xe, dim)
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(Xe))
			stata[i] = func(Xe[nt...])
		end
		print && (@info("Total statistics:"); display([names stata]))
		statas = Matrix{eltype(W[k])}(undef, size(Xe, dim), k)
		for s in 1:k
			Xes = map(W[k][:,s:s] * H[k][s:s,:])
			for i in axes(Xe, dim)
				nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(Xes))
				statas[i, s] = func(vec(Xes[nt...]))
			end
		end
		print && (@info("Signal statistics:"); display([names statas[:, isignalmap]]))
		for s in 1:k
			for i in axes(Xe, dim)
				if stata[i] != 0
					statas[i, s] /= stata[i]
				end
			end
		end
		print && (@info("Normalized signal statistics:"); display([names statas[:, isignalmap]]))
		nm = maximum(statas; dims=2)
		for s in 1:k
			for i in axes(Xe, dim)
				if nm[i] != 0
					statas[i, s] /= nm[i]
				end
			end
		end
		filename = casefilename == "" ? "" : casefilename * "_$(k)signals.$(plotformat)"
		NMFk.plotmatrix(statas[:, isignalmap]; xticks=["S$i" for i=1:k], yticks=names, filename=filename, kw...)
	end
end

function plot_signal_selecton(nkrange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, fitquality::AbstractVector, robustness::AbstractVector, X::AbstractMatrix, W::AbstractVector, H::AbstractVector; figuredir::AbstractString=".", casefilename::AbstractString="signal_selection", title::AbstractString="", xtitle::AbstractString="Number of signals", ytitle::AbstractString="Normalized metrics", plotformat::AbstractString="png", normalize_robustness::Bool=true, plotr2::Bool=true, kw...)
	r = normalize_robustness ? robustness[nkrange] ./ maximumnan(robustness[nkrange]) : robustness[nkrange]
	r2 = similar(robustness)
	mm = maximum(X)
	for k in nkrange
		Xe = W[k] * H[k]
		@assert size(Xe) == size(X)
		r2[k] = NMFk.r2(X, Xe)
		if plotr2
			m = max(mm, maximum(Xe))
			NMFk.plotscatter(X ./ m, Xe ./ m; filename="$(figuredir)/$(casefilename)-$(k)-scatter.$(plotformat)", title="Number of signals = $k R2 = $(round(r2[k]; sigdigits=3))", ymin=0, xmin=0, ymax=1, xmax=1, xtitle="Truth", ytitle="Estimate", quiet=false)
		end
	end
	Mads.plotseries([fitquality[nkrange] ./ maximumnan(fitquality[nkrange]) r r2[nkrange]], "$(figuredir)/$(casefilename)_r2.$(plotformat)"; title=title, ymin=0, xaxis=nkrange, xmin=nkrange[1], xtitle=xtitle, ytitle=ytitle, names=["Fit", "Robustness", "R2"], kw...)
	return r2
end
function plot_signal_selecton(nkrange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, fitquality::AbstractVector, robustness::AbstractVector; figuredir::AbstractString=".", casefilename::AbstractString="signal_selection", title::AbstractString="", xtitle::AbstractString="Number of signals", ytitle::AbstractString="Normalized metrics", plotformat::AbstractString="png", normalize_robustness::Bool=true, kw...)
	r = normalize_robustness ? robustness[nkrange] ./ maximumnan(robustness[nkrange]) : robustness[nkrange]
	Mads.plotseries([fitquality[nkrange] ./ maximumnan(fitquality[nkrange]) r], "$(figuredir)/$(casefilename).$(plotformat)"; title=title, ymin=0, xaxis=nkrange, xmin=nkrange[1], xtitle=xtitle, ytitle=ytitle, names=["Fit", "Robustness"], kw...)
end

plot_feature_selecton = plot_signal_selecton

function showsignals(X::AbstractMatrix, Xnames::AbstractVector; Xmap::AbstractVector=[], order::Function=i->sortperm(i; rev=true), filter_vals::Function=v->findlast(i->i>0.95, v), filter_names=v->occursin.(r".", v))
	local Xm
	if size(X, 1) == length(Xnames)
		Xm = X ./ maximum(X; dims=1)
	elseif size(X, 2) == length(Xnames)
		Xm = permutedims(X ./ maximum(X; dims=2))
	elseif size(X, 1) == length(Xmap)
		mu = unique(Xmap)
		na = length(mu)
		@assert length(Xnames) == na
		Xa = Matrix{eltype(X)}(undef, size(X, 2), na)
		for (i, m) in enumerate(mu)
			Xa[i,:] = sum(X[:, Xmap .== m]; dims=1)
		end
		Xm = Xa ./ maximum(Xa; dims=1)
	elseif size(X, 2) == length(Xmap)
		mu = unique(Xmap)
		na = length(mu)
		@assert length(Xnames) == na
		Xa = Matrix{eltype(X)}(undef, size(X, 1), na)
		for (i, m) in enumerate(mu)
			Xa[:,i] = sum(X[:, Xmap .== m]; dims=2)
		end
		Xm = permutedims(Xa ./ maximum(Xa; dims=2))
	else
		@error("Dimensions do not match!")
		return
	end
	for i in axes(X, 1)
		@info("Signal $i")
		is = order(Xm[:,i])
		ivl = filter_vals(Xm[:,i][is])
		inm = filter_names(Xnames[is][1:ivl])
		display([Xnames[is] Xm[:,i][is]][1:ivl,:][inm,:])
	end
end

# This function selects the indices of the most important signals for plotting, based on a combination of user-specified important names and cluster representatives. It ensures that the number of selected signals does not exceed the specified limit, and prioritizes user-specified important signals when truncation is necessary.
function _select_importance_indexing(ranking::AbstractVector{<:Integer}, names::AbstractVector, important::AbstractVector, labels::AbstractVector, limit::Integer;
		matrix_label::AbstractString, casefilename::AbstractString)
	limit <= 0 && return Int[]
	mandatory = Int[]
	names_str = string.(names)
	if !isempty(important)
		wanted = Set(string.(important))
		for (i, n) in pairs(names_str)
			(n in wanted) && push!(mandatory, i)
		end
		missing = setdiff(wanted, Set(names_str))
		if !isempty(missing)
			@warn("Requested $(matrix_label)_important names not found in $(matrix_label)names: $(collect(missing))")
		end
	end
	for c in unique(labels)
		if (c isa Char) && c == ' '
			continue
		end
		pos = Base.findfirst(i -> labels[i] == c, ranking)
		!isnothing(pos) && push!(mandatory, ranking[pos])
	end
	mandatory = unique(mandatory)
	if length(mandatory) > limit
		@warn("$(matrix_label) ($(casefilename)) required $(matrix_label)_important + cluster representatives ($(length(mandatory))) exceed plot_important_size=$(limit); truncating selection.")
		selected = Int[]
		if !isempty(important)
			wanted = Set(string.(important))
			for (i, n) in pairs(names_str)
				if (n in wanted) && !(i in selected)
					push!(selected, i)
					length(selected) >= limit && break
				end
			end
		end
		if length(selected) < limit
			for idx in ranking
				(idx in mandatory) || continue
				(idx in selected) && continue
				push!(selected, idx)
				length(selected) >= limit && break
			end
		end
		return selected
	end
	selected = copy(mandatory)
	if length(selected) < limit
		for idx in ranking
			(idx in selected) && continue
			push!(selected, idx)
			length(selected) >= limit && break
		end
	end
	return selected
end

function postprocess(W::AbstractMatrix{T}, H::AbstractMatrix{T}, aw...; kw...) where {T <: Number}
	k = size(W, 2)
	@assert size(H, 1) == k
	Wa = Vector{Matrix{T}}(undef, k)
	Ha = Vector{Matrix{T}}(undef, k)
	Wa[k] = W
	Ha[k] = H
	NMFk.postprocess(k, Wa, Ha, aw...; kw...)
end

function postprocess(nkrange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, nruns::Integer; cutoff::Number=0.5, kw...)
	NMFk.postprocess(NMFk.getks(nkrange, silhouette[nkrange], cutoff), nkrange, nruns; kw...)
end

function postprocess(krange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, nkrange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, nruns::Integer; resultdir::AbstractString=".", casefilename::AbstractString="nmfk", suffix::AbstractString="$(casefilename)-$(nruns)", kw...)
	W, H, fit, silhouette, aic, kopt = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename=casefilename)
	NMFk.postprocess(krange, W, H; resultdir="results-$(suffix)", figuredir="figures-$(suffix)", kw...)
end

function postprocess(W::AbstractMatrix, H::AbstractMatrix, aw...; kw...)
	kopt = size(H, 1)
	Wvector = Vector{Matrix}(undef, kopt)
	Hvector = Vector{Matrix}(undef, kopt)
	Wvector[kopt] = W
	Hvector[kopt] = H
	NMFk.postprocess(kopt, Wvector, Hvector, aw...; kw...)
end

"""
    PostprocessOptions

Container for `postprocess` options.

This is intentionally lightweight: it stores only explicitly provided option
values as a `NamedTuple` so callers can keep `postprocess` call sites short.
"""
struct PostprocessOptions{T<:NamedTuple}
	values::T
end

PostprocessOptions() = PostprocessOptions((;))
PostprocessOptions(; kw...) = PostprocessOptions((; kw...))

const _POSTPROCESS_ALLOWED_KEYS = Set{Symbol}([
	:Wnames,
	:Hnames,
	:ordersignals,
	:plot_important_size,
	:Wtimeseries_locations_size,
	:W_important,
	:Htimeseries_locations_size,
	:H_important,
	:clusterW,
	:clusterH,
	:loadassignements,
	:Wsize,
	:Hsize,
	:Wmap,
	:Hmap,
	:Worder,
	:Horder,
	:lon,
	:lat,
	:hover,
	:resultdir,
	:figuredir,
	:Wcasefilename,
	:Hcasefilename,
	:Wtypes,
	:Htypes,
	:Wcolors,
	:Hcolors,
	:dendrogram_color,
	:background_color,
	:createdendrogramsonly,
	:createplots,
	:creatematrixplotsall,
	:createbiplots,
	:createbiplotsall,
	:Wbiplotlabel,
	:Hbiplotlabel,
	:adjustbiplotlabel,
	:biplotlabel,
	:biplotcolor,
	:plottimeseries,
	:plotmaps,
	:plotmap_scope,
	:map_format,
	:map_kw,
	:cutoff,
	:cutoff_s,
	:cutoff_label,
	:Wmatrix_font_size,
	:Hmatrix_font_size,
	:adjustsize,
	:vsize,
	:hsize,
	:W_vsize,
	:W_hsize,
	:H_vsize,
	:H_hsize,
	:Wmatrix_vsize,
	:Wmatrix_hsize,
	:Wdendrogram_vsize,
	:Wdendrogram_hsize,
	:Wtimeseries_vsize,
	:Wtimeseries_hsize,
	:Hmatrix_vsize,
	:Hmatrix_hsize,
	:Hdendrogram_vsize,
	:Hdendrogram_hsize,
	:Htimeseries_vsize,
	:Htimeseries_hsize,
	:Wtimeseries_xaxis,
	:Htimeseries_xaxis,
	:plotmatrixformat,
	:biplotformat,
	:plotseriesformat,
	:sortmag,
	:plotmethod,
	:point_size_nolabel,
	:point_size_label,
	:biplotseparate,
	:biplot_point_label_font_size,
	:repeats,
	:Wrepeats,
	:Hrepeats,
	:movies,
	:quiet,
	:veryquiet,
])

function _postprocess_validate_keys!(merged::NamedTuple)
	bad = Symbol[]
	for k in keys(merged)
		(k in _POSTPROCESS_ALLOWED_KEYS) || push!(bad, k)
	end
	if !isempty(bad)
		throw(ArgumentError("Unsupported postprocess keyword(s): $(join(string.(sort(bad)), ", "))"))
	end
	return nothing
end

"""Postprocess (options as positional argument)."""
function postprocess(options::PostprocessOptions,
		krange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer},
		W::AbstractVector,
		H::AbstractVector,
		X::AbstractMatrix=Matrix{Float32}(undef, 0, 0);
		kw...)
	opt_nt = options.values
	kw_nt = (; kw...)
	# Track explicit overrides when both provide the same key.
	for k in intersect(keys(opt_nt), keys(kw_nt))
		if opt_nt[k] != kw_nt[k]
			@warn("Postprocess option overridden: key=$(k) option_value=$(opt_nt[k]) kw_value=$(kw_nt[k])")
		end
	end
	merged = merge(opt_nt, kw_nt)
	_postprocess_validate_keys!(merged)
	return NMFk.postprocess(krange, W, H, X; merged...)
end

function postprocess(krange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector, X::AbstractMatrix=Matrix{Float32}(undef, 0, 0); Wnames::AbstractVector=["W$i" for i in axes(W[krange[1]], 1)],
		Hnames::AbstractVector=["H$i" for i in axes(H[krange[1]], 2)],
		ordersignals::Symbol=:importance, plot_important_size::Integer=30, Wtimeseries_locations_size::Integer=3, W_important::AbstractVector=[], Htimeseries_locations_size::Integer=3, H_important::AbstractVector=[],
		clusterW::Bool=true, clusterH::Bool=true, loadassignements::Bool=true,
		Wsize::Integer=0, Hsize::Integer=0, Wmap::Union{AbstractVector,AbstractMatrix}=[], Hmap::Union{AbstractVector,AbstractMatrix}=[],
		Worder::AbstractVector=collect(eachindex(Wnames)), Horder::AbstractVector=collect(eachindex(Hnames)),
		lon=nothing, lat=nothing, hover=nothing,
		resultdir::AbstractString=".", figuredir::AbstractString=resultdir,
		Wcasefilename::AbstractString="locations", Hcasefilename::AbstractString="attributes",
		Wtypes::AbstractVector=[], Htypes::AbstractVector=[],
		Wcolors::AbstractVector=NMFk.colors, Hcolors::AbstractVector=NMFk.colors,
		dendrogram_color::AbstractString="black",
		background_color::AbstractString="white",
		createdendrogramsonly::Bool=false, createplots::Bool=!createdendrogramsonly, creatematrixplotsall::Bool=false, createbiplots::Bool=createplots, createbiplotsall::Bool=false,
		Wbiplotlabel::Bool=!(length(Wnames) > 20), Hbiplotlabel::Bool=!(length(Hnames) > 20),
		adjustbiplotlabel::Bool=false, biplotlabel::Symbol=:WH, biplotcolor::Symbol=:WH,
		plottimeseries::Symbol=:none, plotmaps::Bool=false, plotmap_scope::Symbol=:mapbox_contour, map_format::AbstractString="png",
		map_kw::Union{Base.Pairs,NamedTuple,AbstractDict}=Dict(),
		cutoff::Number=0, # cutoff::Number = 0.9 recommended
 		cutoff_s::Number=0, # cutoff_s::Number = 0.95 recommended
		cutoff_label::Number=0.2,
		Wmatrix_font_size::Measures.AbsoluteLength=10Gadfly.pt, Hmatrix_font_size::Measures.AbsoluteLength=10Gadfly.pt,
		adjustsize::Bool=false, vsize::Measures.AbsoluteLength=6Gadfly.inch, hsize::Measures.AbsoluteLength=6Gadfly.inch,
		W_vsize=vsize, W_hsize=hsize, H_vsize=vsize, H_hsize=hsize,
		Wmatrix_vsize=W_vsize, Wmatrix_hsize=W_hsize,
		Wdendrogram_vsize=W_vsize, Wdendrogram_hsize=W_hsize,
		Wtimeseries_vsize=W_vsize, Wtimeseries_hsize=W_hsize,
		Hmatrix_vsize=H_vsize, Hmatrix_hsize=H_hsize,
		Hdendrogram_vsize=H_vsize, Hdendrogram_hsize=H_hsize,
		Htimeseries_vsize=H_vsize, Htimeseries_hsize=H_hsize,
		Wtimeseries_xaxis::AbstractVector=Wnames,
		Htimeseries_xaxis::AbstractVector=Hnames,
		plotmatrixformat="png", biplotformat="pdf", plotseriesformat="png",
		sortmag::Bool=false, plotmethod::Symbol=:frame,
		point_size_nolabel::Measures.AbsoluteLength=3Gadfly.pt, point_size_label::Measures.AbsoluteLength=3Gadfly.pt,
		biplotseparate::Bool=false, biplot_point_label_font_size::Measures.AbsoluteLength=12Gadfly.pt,
		repeats::Integer=1000, Wrepeats::Integer=repeats, Hrepeats::Integer=repeats, movies::Bool=true,
		quiet::Bool=false, veryquiet::Bool=true)
	if length(krange) == 0
		@warn("No optimal solutions")
		return
	end
	@assert length(Wnames) > 0
	@assert length(Hnames) > 0
	@assert length(Wnames) == length(Worder)
	@assert length(Hnames) == length(Horder)
	@assert length(Wtimeseries_xaxis) == length(Wnames)
	@assert length(Htimeseries_xaxis) == length(Hnames)
	@assert any(Worder .=== nothing) == false
	@assert any(Horder .=== nothing) == false
	if map_kw == Dict()
		if plotmap_scope == :well
			map_kw = Dict(:showland=>false, :size=>5, :scale=>2)
		end
	end
	# Adjust biplot labels based on the number of names and the specified biplotlabel option.
	if adjustbiplotlabel
		if length(Wnames) > 100 && length(Hnames) > 100
			biplotlabel = :none
		elseif length(Wnames) > 100
			if biplotlabel == :W
				biplotlabel = :none
			elseif biplotlabel == :WH
				biplotlabel = :H
			end
		elseif length(Hnames) > 100
			if biplotlabel == :H
				biplotlabel = :none
			elseif biplotlabel == :WH
				if length(Wnames) > 100
					biplotlabel = :none
				else
					biplotlabel = :W
				end
			end
		end
	end
	# Generate signal orderings based on the specified method (importance, cluster, or original).
	if length(Htypes) > 0
		if Hcolors == NMFk.colors
			Hcolors = Vector{String}(undef, length(Htypes))
			for (j, t) in enumerate(unique(Htypes))
				Hcolors[Htypes .== t] .= NMFk.colors[j]
			end
		end
		Hnametypes = (Hnames .* " " .* String.(Htypes))[Horder]
	else
		Hnametypes = Hnames[Horder]
	end
	Hnamesmaxlength = max(length.(Hnames)...)
	# Adjust Wnametypes and Wcolors based on Wtypes, ensuring that colors are assigned according to unique types and that names are concatenated with types for labeling.
	if length(Wtypes) > 0
		if Wcolors == NMFk.colors
			Wcolors = Vector{String}(undef, length(Wtypes))
			for (j, t) in enumerate(unique(Wtypes))
				Wcolors[Wtypes .== t] .= NMFk.colors[j]
			end
		end
		Wnametypes = (Wnames .* " " .* String.(Wtypes))[Worder]
	else
		Wnametypes = Wnames[Worder]
	end
	if eltype(Wnames) <: AbstractString
		Wnamesmaxlength = max(length.(Wnames)...)
	elseif eltype(Wnames) <: Dates.AbstractDateTime
		Wnamesmaxlength = max(length.(string.(Wnames))...)
	else
		Wnamesmaxlength = 0
	end
	# Reorder Wnames and Hnames according to Worder and Horder, ensuring that the names are aligned with the order of the signals for subsequent plotting and analysis.
	Wnames = Wnames[Worder]
	Hnames = Hnames[Horder]
	# Validate the lengths of lon and lat coordinates against the lengths of Wnames and Hnames, ensuring that they are compatible for plotting on a map. If the lengths do not match, an error is raised, and if plotting is enabled, an exception is thrown to prevent further execution.
	if plotmaps && !isnothing(lon) && !isnothing(lat)
		if length(lon) == length(lat)
			if length(Hnames) != length(lon) && length(Wnames) != length(lat)
				@error("Length of lon/lat coordinates ($(length(lon))) must be equal to length of either Wnames ($(length(Wnames))) or Hnames ($(length(Hnames)))!")
				if plotmaps || movies
					throw(ErrorException("Length of lon/lat coordinates ($(length(lon))) must be equal to length of either Wnames ($(length(Wnames))) or Hnames ($(length(Hnames)))!"))
				end
			end
		else
			plotmaps = false
			@error("Lat/Lon vector lengths do not match!")
			throw(ErrorException("Lat/Lon vector lengths do not match!"))
		end
	else
		plotmaps = false
	end
	Wclusters = Vector{Vector{Char}}(undef, length(krange))
	Hclusters = Vector{Vector{Char}}(undef, length(krange))
	Sorder = Vector{Vector{Int64}}(undef, length(krange))
	# Generate signal orderings based on the specified method (importance, cluster, or original) for each value of k in krange, and store the results in Wclusters, Hclusters, and Sorder for subsequent analysis and plotting.
	for (ki, k) in enumerate(krange)
		@info("Number of signals: $k")
		if length(X) > 0 # if the input data is provided
			Xe = W[k] * H[k]
			@assert size(Xe) == size(X)
			fitquality = NMFk.normnan(X .- Xe)
			if size(X, 2) < 50
				@info("Relative fits associated with $(Hcasefilename) ...")
				for i in axes(X, 2)
					fitattribute = NMFk.normnan(X[:,i] .- Xe[:,i])
					println("$(Hnames[i]): $(fitattribute/fitquality)")
				end
			end
			if size(X, 1) < 50
				@info("Relative fits associated with $(Wcasefilename) ...")
				for i in axes(X, 1)
					fitattribute = NMFk.normnan(X[i,:] .- Xe[i,:])
					println("$(Wnames[i]): $(fitattribute/fitquality)")
				end
			end
		end

		isignalmap = signalorder(W[k], H[k])

		@info("$(uppercasefirst(Hcasefilename)) (signals=$k)")
		recursivemkdir(resultdir; filename=false)

		if Hsize > 1
			na = convert(Int64, size(H[k], 2) / Hsize)
			Ha = Matrix{eltype(H[k])}(undef, size(H[k], 1), na)
			@assert length(Hnames) == na
			i1 = 1
			i2 = Hsize
			for i = 1:na
				Ha[:,i] = sum(H[k][:,i1:i2]; dims=2)
				i1 += Hsize
				i2 += Hsize
			end
			Ha = Ha[:,Horder]
		elseif size(Hmap, 1) > 0
			@assert size(Hmap, 1) == size(H[k], 2)
			mu = unique(Hmap[:, 1])
			@assert length(Hnames) == length(mu)
			Ha = Matrix{eltype(H[k])}(undef, size(H[k], 1), length(mu))
			for (i, m) in enumerate(mu)
				Ha[:,i] = NMFk.sumnan(H[k][:, Hmap[:, 1] .== m]; dims=2)
			end
			Ha = Ha[:,Horder]
		else
			@assert length(Hnames) == size(H[k], 2)
			Ha = H[k][:,Horder]
		end
		Hmask_nan_cols = vec(all(isnan.(Ha); dims=1))
		if count(Hmask_nan_cols) == size(Ha, 2)
			error("All rows in H matrix are NaN!")
			throw(ErrorException("All rows in H matrix are NaN!"))
		end
		Hm = permutedims(Ha ./ maximum(Ha[:, .!Hmask_nan_cols]; dims=2)) # normalize by rows and PERMUTE (TRANSPOSE)
		Hm[Hm .< eps(eltype(Ha))] .= 0
		Hranking = sortperm(vec(NMFk.sumnan(Hm .^ 2; dims=2)); rev=true) # dims=2 because Hm is already transposed

		# Save the processed H matrix to a CSV file in the specified result directory, including the names of the attributes and the signal labels. The file is named according to the casefilename and the number of signals (k), and is delimited by commas.
		DelimitedFiles.writedlm("$resultdir/Hmatrix-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k))]; Hnames permutedims(Ha)], ',')

		if cutoff > 0
			ia = (Ha ./ maximum(Ha; dims=2)) .> cutoff
			for i in 1:k
				@info("Signal $i (max-normalized elements > $cutoff)")
				display(Hnames[ia[i,:]])
			end
		end

		if Wsize > 1
			na = convert(Int64, size(W[k], 1) / Wsize)
			Wa = Matrix{eltype(W[k])}(undef, na, size(W[k], 2))
			@assert length(Wnames) == na
			i1 = 1
			i2 = Wsize
			for i = 1:na
				Wa[i,:] = NMFk.sumnan(W[k][i1:i2,:]; dims=1)
				i1 += Wsize
				i2 += Wsize
			end
			Wa = Wa[Worder,:]
		elseif size(Wmap, 1) > 0
			@assert size(Wmap, 1) == size(W[k], 1)
			mu = unique(Wmap[:, 1])
			@assert length(Wnames) == length(mu)
			Wa = Matrix{eltype(W[k])}(undef, length(mu), size(W[k], 2))
			for (i, m) in enumerate(mu)
				Wa[i,:] = NMFk.sumnan(W[k][Wmap[:, 1] .== m,:]; dims=1)
			end
			Wa = Wa[Worder,:]
		else
			@assert length(Wnames) == size(W[k], 1)
			Wa = W[k][Worder,:]
		end
		Wmask_nan_rows = vec(all(isnan.(Wa); dims=2))
		if count(Wmask_nan_rows) == size(Wa, 1)
			error("All rows in W matrix are NaN!")
			throw(ErrorException("All rows in W matrix are NaN!"))
		end
		Wm = Wa ./ maximum(Wa[.!Wmask_nan_rows, :]; dims=1) # normalize by columns
		Wm[Wm .< eps(eltype(Wa))] .= 0
		Wranking = sortperm(vec(NMFk.sumnan(Wm .^ 2; dims=2)); rev=true)

		# Define plot sizes based on the number of names and the number of signals
		if (createplots || createdendrogramsonly) && adjustsize
			wr = length(Wnames) / k
			Wmatrix_hsize = Wmatrix_vsize / wr + 3Gadfly.inch
			Wdendrogram_hsize = Wdendrogram_vsize / wr + 5Gadfly.inch
			wr = length(Hnames) / k
			Hmatrix_hsize = Hmatrix_vsize / wr + 3Gadfly.inch
			Hdendrogram_hsize = Hdendrogram_vsize / wr + 5Gadfly.inch
		end

		# Signal importance order based on H clustering
		if clusterH
			reduced = false
			if size(Ha, 1) > 100_000 && Hrepeats > 1
				Hrepeats = 1
				reduced = true
			elseif size(Ha, 1) > 10_000 && Hrepeats > 10
				Hrepeats = 10
				reduced = true
			elseif size(Ha, 1) > 1_000 && Hrepeats > 100
				Hrepeats = 100
				reduced = true
			end
			reduced && @warn("Number of repeats $(Hrepeats) is too high for the matrix size $(size(Ha))! The number of repeats reduced to $(Hrepeats).")
			if count(Hmask_nan_cols) > 0
				@info("Masking NaN cols in H matrix for clustering with average row values...")
				v = NMFk.meannan(Ha[:, .!Hmask_nan_cols]; dims=2)
				Ha[:, Hmask_nan_cols] .= repeat(v, inner=(count(Hmask_nan_cols), 1))
			end
			robustkmeans_results = NMFk.robustkmeans(Ha, k, Hrepeats; resultdir=resultdir, casefilename="Hmatrix", load=loadassignements, save=true, compute_silhouettes_flag=size(Ha, 1) <= 1000)
			H_labels = NMFk.labelassignements(robustkmeans_results.assignments)
			@info("Cluster labels: $(H_labels)")
			if count(Hmask_nan_cols) > 0
				Ha[:, Hmask_nan_cols] .= NaN
			end
			clusterlabels = sort(unique(H_labels))
			@info("Finding best cluster labels ...")
			Hsignalmap = NMFk.signalassignments(Ha, H_labels; clusterlabels=clusterlabels, dims=2)
		end

		# Signal importance order based on W clustering
		if clusterW
			reduced = false
			if size(Wa, 1) > 100_000 && Wrepeats > 1
				Wrepeats = 1
				reduced = true
			elseif size(Wa, 1) > 10_000 && Wrepeats > 10
				Wrepeats = 10
				reduced = true
			elseif size(Wa, 1) > 1_000 && Wrepeats > 100
				Wrepeats = 100
				reduced = true
			end
			reduced && @warn("Number of repeats $(Wrepeats) is too high for the matrix size $(size(Wa))! The number of repeats reduced to $(Wrepeats).")
			if count(Wmask_nan_rows) > 0
				@info("Masking NaN rows in W matrix for clustering with average col values ...")
				v = NMFk.meannan(Wa[.!Wmask_nan_rows, :]; dims=1)
				Wa[Wmask_nan_rows, :] .= repeat(v, inner=(sum(Wmask_nan_rows), 1))
			end
			robustkmeans_results = NMFk.robustkmeans(permutedims(Wa), k, Wrepeats; resultdir=resultdir, casefilename="Wmatrix", load=loadassignements, save=true, compute_silhouettes_flag=size(Wa, 1) <= 1000)
			W_labels = NMFk.labelassignements(robustkmeans_results.assignments)
			if count(Wmask_nan_rows) > 0
				Wa[Wmask_nan_rows, :] .= NaN
			end
			if clusterH
				if clusterlabels != sort(unique(W_labels))
					@warn("W and H cluster labels do not match!")
				end
			else
				clusterlabels = sort(unique(W_labels))
			end
			Wsignalmap = NMFk.signalassignments(Wa, W_labels; clusterlabels=clusterlabels, dims=1)
		end

		# Signal importance order
		if ordersignals == :importance
			@info("Signal importance based on the contribution: $isignalmap")
			signalmap = isignalmap
		elseif ordersignals == :Hcount && clusterH
			@info("Signal importance based on H matrix clustering: $Hsignalmap")
			signalmap = Hsignalmap
		elseif ordersignals == :Wcount && clusterW
			@info("Signal importance based on W matrix clustering: $Wsignalmap")
			signalmap = Wsignalmap
		elseif ordersignals == :none
			@info("No signal importance order requested!")
			signalmap = 1:k
		else
			@warn("Unknown signal order requested $(ordersignals); Signal importance based on the contribution will be used!")
			signalmap = isignalmap
		end
		Sorder[ki] = signalmap

		# W Clustering and plotting
		if clusterH
			Hsignalremap = indexin(signalmap, Hsignalmap)
			cassgined = zeros(Int64, length(Hnames))
			W_labels_new = Vector{eltype(H_labels)}(undef, length(H_labels))
			W_labels_new .= ' '
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(H_labels, [clusterlabels[Hsignalremap[j]]]) .== true
				W_labels_new[ii] .= i
				cassgined[ii] .+= 1
				@info("Signal $(clusterlabels[Hsignalremap[j]]) -> $(i) Count: $(sum(ii))")
			end
			Hclusters[ki] = W_labels_new
			if any(cassgined .== 0)
				@warn("$(uppercasefirst(Hcasefilename)) not assigned to any cluster:")
				display(Hnames[cassgined .== 0])
				@error("Something is wrong!")
			end
			if any(cassgined .> 1)
				@warn("$(uppercasefirst(Hcasefilename)) assigned to more than cluster:")
				display([Hnames[cassgined .> 1] cassgined[cassgined .> 1]])
				@error("Something is wrong!")
			end
			clustermap = Vector{Char}(undef, k)
			clustermap .= ' '
			io = open("$resultdir/$(Hcasefilename)-$(k)-groups.txt", "w")
			for (j, i) in enumerate(clusterlabels)
				@info("Signal $i (S$(signalmap[j])) (k-means clustering)")
				write(io, "Signal $i (S$(signalmap[j]))\n")
				ii = indexin(W_labels_new, [i]) .== true
				is = sortperm(Hm[ii,signalmap[j]]; rev=true)
				d = [Hnames[ii] Hm[ii,signalmap[j]]][is,:]
				display(d)
				for i in axes(d, 1)
					write(io, "$(rpad(d[i,1], Hnamesmaxlength))\t$(round(d[i,2]; sigdigits=3))\n")
				end
				write(io, '\n')
				clustermap[signalmap[j]] = i
			end
			close(io)
			@assert signalmap == sortperm(clustermap)
			@assert clustermap[signalmap] == clusterlabels
			dumpcsv = true
			if plotmaps && length(lon) == length(W_labels_new)
				if isnothing(hover)
					hover = Hnames
				end
				if length(hover) > 1000
					@info("Removing hover text; too many labels $(length(Hnames))!")
					hover = []
				end
				if plotmap_scope == :well
					NMFk.plot_wells("$(Hcasefilename)-$(k)-map.$(map_format)", lon, lat, W_labels_new; figuredir=figuredir, hover=hover, title="Signals: $k")
				elseif plotmap_scope == :mapbox || plotmap_scope == :mapbox_contour
					NMFk.mapbox(lon, lat, W_labels_new; filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, title="Signals: $k", map_kw...)
					NMFk.mapbox(lon, lat, Hm[:,signalmap], clusterlabels; filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, map_kw...)
					if plotmap_scope == :mapbox_contour
						for (i, c) in enumerate(clusterlabels)
							@info("Plotting H map contour for signal $(c) ...")
							NMFk.mapbox_contour(lon, lat, Hm[:,signalmap][:,i]; zmin=0, zmax=1, filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map-contour-signal-$(c).$(map_format)"), location_names=hover, title_colorbar="Signal $(c)", concave_hull=true, show_locations=false, map_kw...)
							if movies && size(Hmap, 2) > 1
								Hm2labels = unique(Hmap[:, 1])
								Hm2bins = unique(Hmap[:, 2])
								@assert length(Hnames) == length(Hm2labels)
								@info("H ($(Hcasefilename)) matrix plot as transient movie ...")
								png_files = Vector{String}(undef, length(Hm2bins))
								hmax = NMFk.maximumnan(H[k]; dims=2)[signalmap]
								for (j, b) in enumerate(Hm2bins)
									@info("Plotting H map contour for signal $(c) bin $(b) ...")
									bin_mask = Hmap[:, 2] .== b
									png_files[j] = joinpath(figuredir, "$(Hcasefilename)-$(k)-map-contour-signal-$(c)-bin-$(b).$(map_format)")
									NMFk.mapbox_contour(lon, lat, H[k][signalmap,bin_mask][i,:] ./ hmax[i]; zmin=0, zmax=1, filename=png_files[j], location_names=hover, title_colorbar="$(b)<br>Signal $(c)", concave_hull=true, show_locations=false, map_kw...)
								end
								NMFk.makemovie(joinpath(figuredir, "$(Hcasefilename)-$(k)-map-contour-signal-$(c)"); files=png_files,cleanup=true)
							end
						end
					end
				else
					NMFk.plotmaps(lon, lat, W_labels_new; filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map.$(map_format)"), title="Signals: $k", scope=string(plotmap_scope), map_kw...)
				end
				DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(clusterlabels) "Signal"]; Hnames lon lat Hm[:,signalmap] W_labels_new], ',')
				dumpcsv = false
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" permutedims(clusterlabels) "Signal"]; Hnames Hm[:,signalmap] W_labels_new], ',')
			end
			yticks = string.(Hnames) .* " " .* string.(W_labels_new)
			if (createdendrogramsonly || createplots || creatematrixplotsall)
				if length(Hranking) > plot_important_size
					@warn("H ($(Hcasefilename)) matrix has too many columns to plot; only plotting top $(plot_important_size) columns!")
					H_importance_indexing = _select_importance_indexing(Hranking, Hnames, H_important, W_labels_new, plot_important_size; matrix_label="H", casefilename=Hcasefilename)
					H_cs_plot = sortperm(W_labels_new[H_importance_indexing])
				else
					@info("H ($(Hcasefilename)) matrix plotting all columns ...")
					H_importance_indexing = Colon()
					H_cs_plot = sortperm(W_labels_new)
				end
				H_plot = Hm[H_importance_indexing,:]
				H_plot[isnan.(H_plot)] .= 0.0
				Hm_col = permutedims(Ha ./ maximum(Ha[:, .!Hmask_nan_cols]; dims=1)) # normalize by cols and PERMUTE (TRANSPOSE)
				H_plot_col = Hm_col[H_importance_indexing,:]
				H_plot_col[H_plot_col .< eps(eltype(H_plot_col))] .= 0
				H_plot_col[isnan.(H_plot_col)] .= 0.0
				if !createdendrogramsonly && creatematrixplotsall
					NMFk.plotmatrix(H_plot; filename="$figuredir/$(Hcasefilename)-$(k)-original.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=yticks[H_importance_indexing], colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					NMFk.plotmatrix(H_plot[:,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks[H_importance_indexing], colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
				end
				if !createdendrogramsonly && createplots
					NMFk.plotmatrix(H_plot[H_cs_plot,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks[H_importance_indexing][H_cs_plot], colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					NMFk.plotmatrix(H_plot_col[H_cs_plot,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted-column.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks[H_importance_indexing][H_cs_plot], colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					if length(Htypes) > 0
						yticks2 = (string.(Hnametypes) .* " " .* string.(W_labels_new))[H_importance_indexing]
						NMFk.plotmatrix(H_plot[:,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-types.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks2, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					end
					if plottimeseries == :H || plottimeseries == :WH
						@info("H ($(Hcasefilename)) matrix timeseries plotting ...")
						Mads.plotseries(Hm, "$figuredir/$(Hcasefilename)-$(k)-timeseries.$(plotseriesformat)"; xaxis=Htimeseries_xaxis, xmin=minimum(Htimeseries_xaxis), xmax=maximum(Htimeseries_xaxis), vsize=Htimeseries_vsize, hsize=Htimeseries_hsize, names=string.(clusterlabels))
						if size(Hmap, 2) > 0
							Hm2labels = unique(Hmap[:, 2])
							Ha2 = Matrix{eltype(H[k])}(undef, length(Hm2labels), size(H[k], 1))
							for (i, m) in enumerate(Hm2labels)
								Ha2[i,:] = NMFk.sumnan(H[k][Hmap[:, 2] .== m,:]; dims=1)
							end
							Hm2 = Ha2 ./ NMFk.maximumnan(Ha2; dims=1) # normalize by columns
							Hm2ranking = sortperm(vec(NMFk.sumnan(Hm2 .^ 2; dims=2)); rev=true)
							@info("H ($(Hcasefilename)) matrix timeseries for specific locations ...")
							for i in Hm2ranking[1:Htimeseries_locations_size]
								println("Plotting timeseries for location: $(Hm2labels[i])")
								well_signals = H[k][Hmap[:,2] .== Hm2labels[i],:]
								if size(well_signals, 1) > 0
									@assert size(well_signals, 1) == length(Htimeseries_xaxis)
									Mads.plotseries(well_signals ./ NMFk.maximumnan(well_signals), "$figuredir/$(Hcasefilename)-$(k)-$(Hm2labels[i])-timeseries.$(plotseriesformat)"; title=string(Hm2labels[i]), xaxis=Htimeseries_xaxis, xmin=minimum(Htimeseries_xaxis), xmax=maximum(Htimeseries_xaxis), vsize=Htimeseries_vsize, hsize=Htimeseries_hsize, names=string.(clusterlabels))
								else
									@warn("No signals found for location $(Hm2labels[i])!")
								end
							end
							@info("H ($(Hcasefilename)) matrix timeseries for specific locations ...")
							for l in H_important
								println("Plotting timeseries for location: $(l)")
								well_signals = H[k][Hmap[:,2] .== l,:]
								if size(well_signals, 1) > 0
									@assert size(well_signals, 1) == length(Htimeseries_xaxis)
									Mads.plotseries(well_signals ./ NMFk.maximumnan(well_signals), "$figuredir/$(Hcasefilename)-$(k)-$(l)-timeseries.$(plotseriesformat)"; title=string(l), xaxis=Htimeseries_xaxis, xmin=minimum(Htimeseries_xaxis), xmax=maximum(Htimeseries_xaxis), vsize=Htimeseries_vsize, hsize=Htimeseries_hsize, names=string.(clusterlabels))
								else
									@warn("No signals found for location $(l)!")
								end
							end
						end
					end
				end
				if (createdendrogramsonly || createplots)
					@info("H ($(Hcasefilename)) matrix dendrogram plotting ...")
					try
						NMFk.plotdendrogram(H_plot[H_cs_plot,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted-dendrogram.$(plotmatrixformat)", metricheat=nothing, xticks=clusterlabels, yticks=yticks[H_importance_indexing][H_cs_plot], minor_label_font_size=Hmatrix_font_size, vsize=Hdendrogram_vsize, hsize=Hdendrogram_hsize, color=dendrogram_color, background_color=background_color, quiet=quiet)
						NMFk.plotdendrogram(H_plot_col[H_cs_plot,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted-dendrogram-column.$(plotmatrixformat)", metricheat=nothing, xticks=clusterlabels, yticks=yticks[H_importance_indexing][H_cs_plot], minor_label_font_size=Hmatrix_font_size, vsize=Hdendrogram_vsize, hsize=Hdendrogram_hsize, color=dendrogram_color, background_color=background_color, quiet=quiet)
					catch errmsg
						!veryquiet && println(errmsg)
						@warn("H ($(Hcasefilename)) matrix dendrogram plotting failed!")
					end
				end
			end
			if createbiplots
				@info("Biplotting H ($(Hcasefilename)) matrix ...")
				createbiplotsall && NMFk.biplots(Hm, Hnames, collect(1:k); smartplotlabel=true, filename="$figuredir/$(Hcasefilename)-$(k)-biplots-original.$(biplotformat)", background_color=background_color, types=W_labels_new, plotlabel=Hbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				NMFk.biplots(Hm[H_cs_plot,signalmap], Hnames[H_cs_plot], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Hcasefilename)-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, types=W_labels_new[H_cs_plot], plotlabel=Hbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				length(Htypes) > 0 && NMFk.biplots(Hm[H_cs_plot,signalmap], Hnames[H_cs_plot], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Hcasefilename)-$(k)-biplots-type.$(biplotformat)", background_color=background_color, colors=Hcolors[H_cs_plot], plotlabel=Hbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
			end
		end

		# W signal importance and clustering results
		@info("$(uppercasefirst(Wcasefilename)) (signals=$k)")
		if cutoff > 0 # if a cutoff is specified, identify and display the names of the attributes in W that have a maximum-normalized value greater than the cutoff for each signal. This helps to highlight the most important attributes associated with each signal based on their contribution to the W matrix.
			ia = (Wa ./ maximum(Wa; dims=1)) .> cutoff
			for i in 1:k
				@info("Signal $i (max-normalized elements > $cutoff)")
				display(Wnames[ia[:,i]])
			end
		end
		# Save the processed W matrix to a CSV file in the specified result directory, including the names of the attributes and the signal labels. The file is named according to the casefilename and the number of signals (k), and is delimited by commas.
		DelimitedFiles.writedlm("$resultdir/Wmatrix-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k))]; Wnames Wa], ',')

		# If W clustering is performed, remap the cluster labels to the signal importance order and save the results to a text file. Additionally, if plotting of the map is requested and the longitude and latitude data are available, create a map visualization of the clusters.
		if clusterW
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(W_labels, [i]) .== true
				@info("Signal $i (S$(Wsignalmap[j])) Count: $(sum(ii))")
			end
			Wsignalremap = indexin(signalmap, Wsignalmap)
			cassgined = zeros(Int64, length(Wnames))
			W_labels_new = Vector{eltype(W_labels)}(undef, length(W_labels))
			W_labels_new .= ' '
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(W_labels, [clusterlabels[Wsignalremap[j]]]) .== true
				W_labels_new[ii] .= i
				cassgined[ii] .+= 1
				@info("Signal $(clusterlabels[Wsignalremap[j]]) -> $(i) Count: $(sum(ii))")
			end
			Wclusters[ki] = W_labels_new
			if any(cassgined .== 0)
				@warn("$(uppercasefirst(Wcasefilename)) not assigned to any cluster:")
				display(Wnames[cassgined .== 0])
				@error("Something is wrong!")
			end
			if any(cassgined .> 1)
				@warn("$(uppercasefirst(Wcasefilename)) assigned to more than cluster:")
				display([Wnames[cassgined .> 1] cassgined[cassgined .> 1]])
				@error("Something is wrong!")
			end
			io = open("$resultdir/$(Wcasefilename)-$(k)-groups.txt", "w")
			for (j, i) in enumerate(clusterlabels)
				@info("Signal $i (remapped k-means clustering)")
				write(io, "Signal $i (remapped k-means clustering)\n")
				ii = indexin(W_labels_new, [i]) .== true
				is = sortperm(Wm[ii,signalmap[j]]; rev=true)
				d = [Wnames[ii] Wm[ii,signalmap[j]]][is,:]
				display(d)
				if Wnamesmaxlength > 0
					for i in axes(d, 1)
						write(io, "$(rpad(d[i,1], Wnamesmaxlength))\t$(round(d[i,2]; sigdigits=3))\n")
					end
				else
					for i in axes(d, 1)
						write(io, "$(d[i,1])\t$(round(d[i,2]; sigdigits=3))\n")
					end

				end
				write(io, '\n')
			end
			close(io)
			dumpcsv = true
			if plotmaps && length(lon) == length(W_labels_new)
				if isnothing(hover)
					hover = Wnames
				end
				if length(hover) > 1000
					@info("Removing hover text; too many labels $(length(Wnames))!")
					hover = []
				end
				if plotmap_scope == :well
					NMFk.plot_wells("$(Wcasefilename)-$(k)-map.$(map_format)", lon, lat, W_labels_new; figuredir=figuredir, hover=hover, title="Signals: $k")
				elseif plotmap_scope == :mapbox || plotmap_scope == :mapbox_contour
					NMFk.mapbox(lon, lat, W_labels_new; filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, title="Signals: $k", map_kw...)
					NMFk.mapbox(lon, lat, Wm[:,signalmap], clusterlabels; filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, map_kw...)
					if plotmap_scope == :mapbox_contour
						for (i, c) in enumerate(clusterlabels)
							@info("Plotting W map contour for signal $(c) ...")
							NMFk.mapbox_contour(lon, lat, Wm[:,signalmap][:,i]; zmin=0, zmax=1, filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map-contour-signal-$(c).$(map_format)"), location_names=hover, title_colorbar="Signal $(c)", concave_hull=true, map_kw...)
							if movies && size(Wmap, 2) > 1
								Wm2labels = unique(Wmap[:, 1])
								Wm2bins = unique(Wmap[:, 2])
								@assert length(Wnames) == length(Wm2labels)
								@info("W ($(Wcasefilename)) matrix plot as transient movie ...")
								png_files = Vector{String}(undef, length(Wm2bins))
								wmax = NMFk.maximumnan(W[k]; dims=1)[signalmap]
								for (j, b) in enumerate(Wm2bins)
									@info("Plotting W map contour for signal $(c) bin $(b) ...")
									bin_mask = Wmap[:, 2] .== b
									png_files[j] = joinpath(figuredir, "$(Wcasefilename)-$(k)-map-contour-signal-$(c)-bin-$(b).$(map_format)")
									NMFk.mapbox_contour(lon, lat, W[k][bin_mask,signalmap][:,i] ./ wmax[i]; zmin=0, zmax=1, filename=png_files[j], location_names=hover, title_colorbar="$(b)<br>Signal $(c)", concave_hull=true, map_kw...)
								end
								NMFk.makemovie(joinpath(figuredir, "$(Wcasefilename)-$(k)-map-contour-signal-$(c)"); files=png_files, cleanup=true)
							end
						end
					end
				else
					NMFk.plotmaps(lon, lat, W_labels_new; filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map.$(map_format)"), title="Signals: $k", scope=string(plotmap_scope), map_kw...)
				end
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(clusterlabels) "Signal"]; Wnames lon lat Wm[:,signalmap] W_labels_new], ',')
				dumpcsv = false
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" permutedims(clusterlabels) "Signal"]; Wnames Wm[:,signalmap] W_labels_new], ',')
			end
			if !isnothing(lon) && (length(lon) != length(W_labels_new)) && (length(lon) != length(W_labels_new))
				@warn("Length of lat/lon coordinates ($(length(lon))) does not match the number of either W matrix rows ($(length(W_labels_new))) or H matrix columns ($(length(W_labels_new)))!")
			end
			yticks = string.(Wnames) .* " " .* string.(W_labels_new)
			if (createdendrogramsonly || createplots || creatematrixplotsall)
				if length(Wranking) > plot_important_size
					@warn("W ($(Wcasefilename)) matrix has too many rows to plot; selecting $(plot_important_size) rows (ensuring one per cluster label) ...")
					W_importance_indexing = _select_importance_indexing(Wranking, Wnames, W_important, W_labels_new, plot_important_size; matrix_label="W", casefilename=Wcasefilename)
					W_cs_plot = sortperm(W_labels_new[W_importance_indexing])
				else
					@info("W ($(Wcasefilename)) matrix plotting all rows ...")
					W_importance_indexing = Colon()
					W_cs_plot = sortperm(W_labels_new)
				end
				W_plot = Wm[W_importance_indexing,:]
				W_plot[isnan.(W_plot)] .= 0.0
				Wm_row = Wa ./ maximum(Wa[.!Wmask_nan_rows, :]; dims=2) # normalize by rows
				W_plot_row = Wm_row[W_importance_indexing,:]
				W_plot_row[W_plot_row .< eps(eltype(W_plot_row))] .= 0
				W_plot_row[isnan.(W_plot_row)] .= 0.0
				if !createdendrogramsonly && creatematrixplotsall
					NMFk.plotmatrix(W_plot; filename="$figuredir/$(Wcasefilename)-$(k)-original.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=yticks[W_importance_indexing], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color)
					# sorted by Wa magnitude
					# ws = sortperm(vec(sum(W_plot; dims=1)); rev=true)
					# NMFk.plotmatrix(W_plot[:,ws]; filename="$figuredir/$(Wcasefilename)-$(k)-original-sorted.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=eachindex(cw)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					cws = sortperm(W_labels[W_importance_indexing])
					yticks3 = ["$(Wnames[i]) $(W_labels[i])" for i=eachindex(W_labels)][W_importance_indexing][cws]
					NMFk.plotmatrix(W_plot[cws,:]; filename="$figuredir/$(Wcasefilename)-$(k)-original-sorted.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=yticks3, colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
					NMFk.plotmatrix(W_plot[:,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks[W_importance_indexing], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
				end
				if !createdendrogramsonly && createplots
					if length(Wtypes) > 0
						yticks2 = (string.Wnametypes .* " " .* W_labels_new)[W_importance_indexing]
						NMFk.plotmatrix(W_plot[:,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-types.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks2, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, quiet=quiet)
					end
					NMFk.plotmatrix(W_plot[W_cs_plot,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks[W_importance_indexing][W_cs_plot], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
					NMFk.plotmatrix(W_plot_row[W_cs_plot,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted-row.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks[W_importance_indexing][W_cs_plot], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
					# NMFk.plotmatrix(W_plot./sum(W_plot; dims=1); filename="$figuredir/$(Wcasefilename)-$(k)-sum.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=eachindex(cols)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					# NMFk.plotmatrix((W_plot./sum(W_plot; dims=1))[W_cs_plot,:]; filename="$figuredir/$(Wcasefilename)-$(k)-sum2.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[W_cs_plot][i]) $(cw[W_cs_plot][i])" for i=eachindex(cols)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					# NMFk.plotmatrix((W_plot ./ sum(W_plot; dims=1))[W_cs_plot,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-labeled-sorted-sumrows.$(plotmatrixformat)", xticks=clusterlabels, yticks=["$(Wnames[W_cs_plot][i]) $(cwnew[W_cs_plot][i])" for i=eachindex(cwnew)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					if plottimeseries == :W || plottimeseries == :WH
						@info("W ($(Wcasefilename)) matrix timeseries plotting ...")
						Mads.plotseries(Wm, "$figuredir/$(Wcasefilename)-$(k)-timeseries.$(plotseriesformat)"; xaxis=Wtimeseries_xaxis, xmin=minimum(Wtimeseries_xaxis), xmax=maximum(Wtimeseries_xaxis), vsize=Wtimeseries_vsize, hsize=Wtimeseries_hsize, names=string.(clusterlabels))
						if size(Wmap, 2) > 1
							Wm2labels = unique(Wmap[:, 2])
							Wa2 = Matrix{eltype(W[k])}(undef, length(Wm2labels), size(W[k], 2))
							for (i, m) in enumerate(Wm2labels)
								Wa2[i,:] = NMFk.sumnan(W[k][Wmap[:, 2] .== m,:]; dims=1)
							end
							Wm2 = Wa2 ./ NMFk.maximumnan(Wa2; dims=1) # normalize by columns
							Wm2ranking = sortperm(vec(NMFk.sumnan(Wm2 .^ 2; dims=2)); rev=true)
							@info("W ($(Wcasefilename)) matrix timeseries plotting for $(Wtimeseries_locations_size) important locations ...")
							for i in Wm2ranking[1:Wtimeseries_locations_size]
								println("Plotting location $(Wm2labels[i]) ...")
								well_signals = W[k][Wmap[:,2] .== Wm2labels[i],:]
								if size(well_signals, 1) > 0
									@assert size(well_signals, 1) == length(Wtimeseries_xaxis)
									Mads.plotseries(well_signals ./ NMFk.maximumnan(well_signals), "$figuredir/$(Wcasefilename)-$(k)-$(Wm2labels[i])-timeseries.$(plotseriesformat)"; title=string(Wm2labels[i]), xaxis=Wtimeseries_xaxis, xmin=minimum(Wtimeseries_xaxis), xmax=maximum(Wtimeseries_xaxis), vsize=Wtimeseries_vsize, hsize=Wtimeseries_hsize, names=string.(clusterlabels))
								else
									@warn("No signals found for location $(Wm2labels[i])!")
								end
							end
							@info("W ($(Wcasefilename)) matrix timeseries plotting for specificly requested locations ...")
							for l in W_important
								println("Plotting location $(l) ...")
								well_signals = W[k][Wmap[:,2] .== l,:]
								if size(well_signals, 1) > 0
									@assert size(well_signals, 1) == length(Wtimeseries_xaxis)
									Mads.plotseries(well_signals ./ NMFk.maximumnan(well_signals), "$figuredir/$(Wcasefilename)-$(k)-$(l)-timeseries.$(plotseriesformat)"; title=string(l), xaxis=Wtimeseries_xaxis, xmin=minimum(Wtimeseries_xaxis), xmax=maximum(Wtimeseries_xaxis), vsize=Wtimeseries_vsize, hsize=Wtimeseries_hsize, names=string.(clusterlabels))
								else
									@warn("No signals found for location $(l)!")
								end
							end
						end
					end
				end
				if createdendrogramsonly || createplots
					try
						@info("W ($(Wcasefilename)) matrix dendrogram plotting ...")
						NMFk.plotdendrogram(W_plot[W_cs_plot,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted-dendrogram.$(plotmatrixformat)", metricheat=nothing, xticks=clusterlabels, yticks=yticks[W_importance_indexing][W_cs_plot], minor_label_font_size=Wmatrix_font_size, vsize=Wdendrogram_vsize, hsize=Wdendrogram_hsize, color=dendrogram_color, background_color=background_color, quiet=quiet)
						NMFk.plotdendrogram(W_plot_row[W_cs_plot,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted-dendrogram-row.$(plotmatrixformat)", metricheat=nothing, xticks=clusterlabels, yticks=yticks[W_importance_indexing][W_cs_plot], minor_label_font_size=Wmatrix_font_size, vsize=Wdendrogram_vsize, hsize=Wdendrogram_hsize, color=dendrogram_color, background_color=background_color, quiet=quiet)
					catch errmsg
						!veryquiet && println(errmsg)
						@warn("W ($(Wcasefilename)) matrix dendrogram plotting failed!")
					end
				end
			end
			if createbiplots
				@info("Biplotting W ($(Wcasefilename)) matrix matrix ...")
				createbiplotsall && NMFk.biplots(Wm, Wnames, collect(1:k); smartplotlabel=true, filename="$figuredir/$(Wcasefilename)-$(k)-biplots-original.$(biplotformat)", background_color=background_color, types=W_labels_new, plotlabel=Wbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				NMFk.biplots(Wm[W_cs_plot,signalmap], Wnames[W_cs_plot], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Wcasefilename)-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, types=W_labels_new[W_cs_plot], plotlabel=Wbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				length(Wtypes) > 0 && NMFk.biplots(Wm[W_cs_plot,signalmap], Wnames[W_cs_plot], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Wcasefilename)-$(k)-biplots-type.$(biplotformat)", background_color=background_color, colors=Wcolors[W_cs_plot], plotlabel=Wbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
			end
			if createbiplots && createbiplotsall
				@info("Biplotting combined W and H matrices ...")
				if biplotlabel == :W
					biplotlabels = [Wnames_label; fill("", length(Hnames))]
					biplotlabelflag = true
				elseif biplotlabel == :WH
					biplotlabels = [Wnames_label; Hnames_label]
					biplotlabelflag = true
				elseif biplotlabel == :H
					biplotlabels = [fill("", length(Wnames)); Hnames_label]
					biplotlabelflag = true
				elseif biplotlabel == :none
					biplotlabels = [fill("", length(Wnames)); fill("", length(Hnames))]
					biplotlabelflag = false
				end
				Wbiplottypecolors = length(Wtypes) > 0 ? Wcolors : set_typecolors(W_labels_new, Wcolors)
				Hbiplottypecolors = length(Htypes) > 0 ? Hcolors : set_typecolors(W_labels_new, Hcolors)
				if biplotcolor == :W
					biplotcolors = [Wbiplottypecolors; fill("gray", length(Hnames))]
				elseif biplotcolor == :WH
					Hbiplottypecolors = length(Htypes) > 0 ? Hcolors : set_typecolors(W_labels_new, Hcolors[k+1:end])
					biplotcolors = [Wbiplottypecolors; Hbiplottypecolors]
				elseif biplotcolor == :H
					biplotcolors = [fill("gray", length(Wnames)); Hbiplottypecolors]
				elseif biplotcolor == :none
					biplotcolors = [fill("blue", length(Wnames)); fill("red", length(Hnames))]
				end
				NMFk.biplots(M, biplotlabels, collect(1:k); smartplotlabel=true, filename="$figuredir/all-$(k)-biplots-original.$(biplotformat)", background_color=background_color, typecolors=biplotcolors, plotlabel=biplotlabelflag, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				if biplotcolor == :W
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,signalmap]
				elseif biplotcolor == :WH
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,signalmap]
				elseif biplotcolor == :H
					M = [permutedims(Ha ./ maximum(Ha)); Wa ./ maximum(Wa)][:,signalmap]
				elseif biplotcolor == :none
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,signalmap]
				end
				NMFk.biplots(M, biplotlabels, clusterlabels; smartplotlabel=true, filename="$figuredir/all-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, typecolors=biplotcolors, plotlabel=biplotlabelflag, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
			end
		end
		# generate association tables
		if cutoff_s > 0
			attributesl = Wsize > 1 ? repeat(Wnames; inner=Wsize) : Wnames
			Xe = W[k] * Ha
			local table = Hnames
			local table2 = Hnames
			local table3 = Hnames
			for i = 1:k
				Xek = (W[k][:,i:i] * Ha[i:i,:]) ./ Xe
				Xekm = Xek .> cutoff_s
				o = findmax(Xek; dims=1)
				table = hcat(table, map(i->attributesl[i], map(i->o[2][i][1], eachindex(Hnames))))
				table2 = hcat(table2, map(i->attributesl[Xekm[:,i]], eachindex(Hnames)))
				table3 = hcat(table3, map(i->sum(Xekm[:,i]), eachindex(Hnames)))
			end
			if !isnothing(lon) && !isnothing(lat)
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_max.csv", [lon lat table], ',')
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_$(cutoff_s).csv", [lon lat table2], ';')
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_count_$(cutoff_s).csv", [lon lat table3], ',')
			else
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_max.csv", table, ',')
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_$(cutoff_s).csv", table2, ';')
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_count_$(cutoff_s).csv", table3, ',')
			end
			local table = attributesl
			local table2 = attributesl
			local table3 = attributesl
			for i = 1:k
				Xek = (W[k][:,i:i] * Ha[i:i,:]) ./ Xe
				Xekm = Xek .> cutoff_s
				o = findmax(Xek; dims=2)
				table = hcat(table, map(i->Hnames[i], map(i->o[2][i][2], eachindex(attributesl))))
				table2 = hcat(table2, map(i->Hnames[Xekm[i,:]], eachindex(attributesl)))
				table3 = hcat(table3, map(i->sum(Xekm[i,:]), eachindex(attributesl)))
			end
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k)-table_max.csv", table, ',')
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k)-table_$(cutoff_s).csv", table2, ';')
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k)-table_count_$(cutoff_s).csv", table3, ',')
		end
	end
	return Sorder, Wclusters, Hclusters
end

function getmissingattributes(X::AbstractMatrix, attributes::AbstractVector, locationclusters::AbstractVector; locationmatrix::Union{Nothing,AbstractMatrix}=nothing, attributematrix::Union{Nothing,AbstractMatrix}=nothing, dims::Integer=2, plothistogram::Bool=false, quiet::Bool=true)
	for (ic, c) in enumerate(unique(sort(locationclusters)))
		i = locationclusters .== c
		@info("Location cluster: $c")
		min, max, std, count = NMFk.datanalytics(X[i,:], attributes; dims=dims, plothistogram=plothistogram, quiet=quiet)
		@info("Missing attribute measurements:")
		if isnothing(attributematrix)
			display(attributes[count.==0])
		else
			p = attributematrix[ic,count.==0]
			is = sortperm(p; rev=true)
			display([attributes[count.==0] p][is,:])
		end
	end
end

clusterresults = postprocess