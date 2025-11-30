import DelimitedFiles
import PlotlyJS
import Mads

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
	Hclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(H, k, Wrepeats; resuldir=resultdir, casefilename=Hclusterlabelcasefilename, load=loadassignements, save=true, silhouettes_flag=false).assignments)
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
	Wclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(permutedims(W), k, repeats; resultdir=resultdir, casefilename=Wclusterlabelcasefilename, load=loadassignements, save=true, silhouettes_flag=false).assignments)
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

function postprocess(krange::Union{AbstractUnitRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector, X::AbstractMatrix=Matrix{Float32}(undef, 0, 0); Wnames::AbstractVector=["W$i" for i in axes(W[krange[1]], 1)],
		Hnames::AbstractVector=["H$i" for i in axes(H[krange[1]], 2)],
		ordersignals::Symbol=:importance,
		clusterW::Bool=true, clusterH::Bool=true, loadassignements::Bool=true,
		Wsize::Integer=0, Hsize::Integer=0, Wmap::AbstractVector=[], Hmap::AbstractVector=[],
		Worder::AbstractVector=collect(eachindex(Wnames)), Horder::AbstractVector=collect(eachindex(Hnames)),
		lon=nothing, lat=nothing, hover=nothing,
		resultdir::AbstractString=".", figuredir::AbstractString=resultdir,
		Wcasefilename::AbstractString="locations", Hcasefilename::AbstractString="attributes",
		Wtypes::AbstractVector=[], Htypes::AbstractVector=[],
		Wcolors::AbstractVector=NMFk.colors, Hcolors::AbstractVector=NMFk.colors,
		dendrogram_color::AbstractString="black",
		background_color::AbstractString="white",
		createdendrogramsonly::Bool=false, createplots::Bool=!createdendrogramsonly, createbiplots::Bool=createplots,
		Wbiplotlabel::Bool=!(length(Wnames) > 20), Hbiplotlabel::Bool=!(length(Hnames) > 20),
		adjustbiplotlabel::Bool=false, biplotlabel::Symbol=:WH, biplotcolor::Symbol=:WH,
		plottimeseries::Symbol=:none, plotmap_scope::Symbol=:mapbox, map_format::AbstractString="png",
		map_kw::Union{Base.Pairs,AbstractDict}=Dict(),
		cutoff::Number=0, # cutoff::Number = 0.9 recommended
 		cutoff_s::Number=0, # cutoff_s::Number = 0.95 recommended
		cutoff_label::Number=0.2,
		Wmatrix_font_size=10Gadfly.pt, Hmatrix_font_size=10Gadfly.pt,
		adjustsize::Bool=false, vsize=6Gadfly.inch, hsize=6Gadfly.inch,
		W_vsize=vsize, W_hsize=hsize, H_vsize=vsize, H_hsize=hsize,
		Wmatrix_vsize=W_vsize, Wmatrix_hsize=W_hsize,
		Wdendrogram_vsize=W_vsize, Wdendrogram_hsize=W_hsize,
		Wtimeseries_vsize=W_vsize, Wtimeseries_hsize=W_hsize,
		Hmatrix_vsize=H_vsize, Hmatrix_hsize=H_hsize,
		Hdendrogram_vsize=H_vsize, Hdendrogram_hsize=H_hsize,
		Htimeseries_vsize=H_vsize, Htimeseries_hsize=H_hsize,
		plotmatrixformat="png", biplotformat="pdf", plotseriesformat="png",
		sortmag::Bool=false, plotmethod::Symbol=:frame,
		point_size_nolabel=3Gadfly.pt, point_size_label=3Gadfly.pt,
		biplotseparate::Bool=false, biplot_point_label_font_size=12Gadfly.pt,
		repeats::Integer=1000, Wrepeats::Integer=repeats, Hrepeats::Integer=repeats,
		quiet::Bool=false, veryquiet::Bool=true)
	if length(krange) == 0
		@warn("No optimal solutions")
		return
	end
	@assert length(Wnames) > 0
	@assert length(Hnames) > 0
	@assert length(Wnames) == length(Worder)
	@assert length(Hnames) == length(Horder)
	@assert any(Worder .=== nothing) == false
	@assert any(Horder .=== nothing) == false
	if map_kw == Dict()
		if plotmap_scope == :well
			map_kw = Dict(:showland=>false, :size=>5, :scale=>2)
		end
	end
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
	Wnames = Wnames[Worder]
	Hnames = Hnames[Horder]
	if !isnothing(lon) && !isnothing(lat)
		if length(lon) == length(lat)
			if length(Hnames) != length(lon) && length(Wnames) != length(lat)
				plotmap = false
				@error("Length of lon/lat coordinates ($(length(lon))) must be equal to length of Wnames ($(length(Wnames))) or Hnames ($(length(Hnames)))!")
			else
				plotmap = true
			end
		else
			plotmap = false
			@error("Lat/Lon vector lengths do not match!")
			return
		end
	else
		plotmap = false
	end
	Wclusters = Vector{Vector{Char}}(undef, length(krange))
	Hclusters = Vector{Vector{Char}}(undef, length(krange))
	Sorder = Vector{Vector{Int64}}(undef, length(krange))
	for (ki, k) in enumerate(krange)
		@info("Number of signals: $k")

		if length(X) > 0 # if the input data is provided
			Xe = W[k] * H[k]
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
		elseif length(Hmap) > 0
			@assert length(Hmap) == size(H[k], 2)
			mu = unique(Hmap)
			na = length(mu)
			@assert length(Hnames) == na
			Ha = Matrix{eltype(H[k])}(undef, size(H[k], 1), na)
			for (i, m) in enumerate(mu)
				Ha[:,i] = sum(H[k][:, Hmap .== m]; dims=2)
			end
			Ha = Ha[:,Horder]
		else
			@assert length(Hnames) == size(H[k], 2)
			Ha = H[k][:,Horder]
		end
		Hm = permutedims(Ha ./ maximum(Ha; dims=2))
		Hm[Hm .< eps(eltype(Ha))] .= 0

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
				Wa[i,:] = sum(W[k][i1:i2,:]; dims=1)
				i1 += Wsize
				i2 += Wsize
				end
				Wa = Wa[Worder,:]
			elseif length(Wmap) > 0
				@assert length(Wmap) == size(W[k], 1)
				mu = unique(Wmap)
				na = length(mu)
				@assert length(Wnames) == na
				Wa = Matrix{eltype(W[k])}(undef, na, size(W[k], 2))
				for (i, m) in enumerate(mu)
					Wa[i,:] = sum(W[k][Wmap .== m,:]; dims=1)
				end
				Wa = Wa[Worder,:]
		else
			@assert length(Wnames) == size(W[k], 1)
			Wa = W[k][Worder,:]
		end
		Wm = Wa ./ maximum(Wa; dims=1)
		Wm[Wm .< eps(eltype(Wa))] .= 0

		if (createplots || createdendrogramsonly) && adjustsize
			wr = length(Wnames) / k
			Wmatrix_hsize = Wmatrix_vsize / wr + 3Gadfly.inch
			Wdendrogram_hsize = Wdendrogram_vsize / wr + 5Gadfly.inch
			wr = length(Hnames) / k
			Hmatrix_hsize = Hmatrix_vsize / wr + 3Gadfly.inch
			Hdendrogram_hsize = Hdendrogram_vsize / wr + 5Gadfly.inch
		end

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
			reduced && @warn("Number of repeats $(Wrepeats) is too high for the matrix size $(size(Wa))! The number of repeats reduced to $(Wrepeats).")
			ch = NMFk.labelassignements(NMFk.robustkmeans(Ha, k, Hrepeats; resultdir=resultdir, casefilename="Hmatrix", load=loadassignements, save=true, silhouettes_flag=false).assignments)
			clusterlabels = sort(unique(ch))
			hsignalmap = NMFk.signalassignments(Ha, ch; clusterlabels=clusterlabels, dims=2)
		end

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
			cw = NMFk.labelassignements(NMFk.robustkmeans(permutedims(Wa), k, Wrepeats; resultdir=resultdir, casefilename="Wmatrix", load=loadassignements, save=true, silhouettes_flag=false).assignments)
			if clusterH
				@assert clusterlabels == sort(unique(cw))
			else
				clusterlabels = sort(unique(cw))
			end
			wsignalmap = NMFk.signalassignments(Wa, cw; clusterlabels=clusterlabels, dims=1)
		end

		if ordersignals == :importance
			@info("Signal importance based on the contribution: $isignalmap")
			signalmap = isignalmap
		elseif ordersignals == :Hcount && clusterH
			@info("Signal importance based on H matrix clustering: $hsignalmap")
			signalmap = hsignalmap
		elseif ordersignals == :Wcount && clusterW
			@info("Signal importance based on W matrix clustering: $wsignalmap")
			signalmap = wsignalmap
		elseif ordersignals == :none
			@info("No signal importance order requested!")
			signalmap = 1:k
		else
			@warn("Unknown signal order requested $(ordersignals); Signal importance based on the contribution will be used!")
			signalmap = isignalmap
		end
		Sorder[ki] = signalmap

		if clusterH
			hsignalremap = indexin(signalmap, hsignalmap)
			cassgined = zeros(Int64, length(Hnames))
			chnew = Vector{eltype(ch)}(undef, length(ch))
			chnew .= ' '
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(ch, [clusterlabels[hsignalremap[j]]]) .== true
				chnew[ii] .= i
				cassgined[ii] .+= 1
				@info("Signal $(clusterlabels[hsignalremap[j]]) -> $(i) Count: $(sum(ii))")
			end
			Hclusters[ki] = chnew
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
				ii = indexin(chnew, [i]) .== true
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
			if plotmap && length(lon) == length(chnew)
				if isnothing(hover)
					hover = Hnames
				end
				if plotmap_scope == :well
					NMFk.plot_wells("$(Hcasefilename)-$(k)-map.$(map_format)", lon, lat, chnew; figuredir=figuredir, hover=hover, title="Signals: $k")
				elseif plotmap_scope == :mapbox
					NMFk.mapbox(lon, lat, chnew; filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, title="Signals: $k", map_kw...)
					NMFk.mapbox(lon, lat, Hm[:,signalmap], clusterlabels; filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, map_kw...)
				else
					NMFk.plotmap(lon, lat, chnew; filename=joinpath(figuredir, "$(Hcasefilename)-$(k)-map.$(map_format)"), title="Signals: $k", scope=string(plotmap_scope), map_kw...)
				end
				DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(clusterlabels) "Signal"]; Hnames lon lat Hm[:,signalmap] chnew], ',')
				dumpcsv = false
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" permutedims(clusterlabels) "Signal"]; Hnames Hm[:,signalmap] chnew], ',')
			end
			cs = sortperm(chnew)
			if createplots
				if length(chnew) >= 100
					@warn("H ($(Hcasefilename)) matrix has too many attributes to plot as a matrix!")
				else
					xticks = ["S$i" for i=1:k]
					yticks = ["$(Hnames[i]) $(chnew[i])" for i=eachindex(chnew)]
					NMFk.plotmatrix(Hm; filename="$figuredir/$(Hcasefilename)-$(k)-original.$(plotmatrixformat)", xticks=xticks, yticks=yticks, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					NMFk.plotmatrix(Hm[:,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					if length(Htypes) > 0
						yticks = ["$(Hnametypes[i]) $(chnew[i])" for i=eachindex(chnew)]
						NMFk.plotmatrix(Hm[:,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-types.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
					end
					yticks = ["$(Hnames[cs][i]) $(chnew[cs][i])" for i=eachindex(chnew)]
					NMFk.plotmatrix(Hm[cs,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Hmatrix_vsize, hsize=Hmatrix_hsize, background_color=background_color, quiet=quiet)
				end
				if plottimeseries == :H || plottimeseries == :WH
					Mads.plotseries(Hm, "$figuredir/$(Hcasefilename)-$(k)-timeseries.$(plotseriesformat)"; xaxis=Hnames, xmin=minimum(Hnames), xmax=maximum(Hnames), vsize=Htimeseries_vsize, hsize=Htimeseries_hsize)
				end
			end
			if (createdendrogramsonly || createplots) && length(chnew) < 100
				@info("Dendrogram ploting ...")
				try
					yticks = ["$(Hnames[cs][i]) $(chnew[cs][i])" for i=eachindex(chnew)]
					NMFk.plotdendrogram(Hm[cs,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted-dendrogram.$(plotmatrixformat)", metricheat=nothing, xticks=clusterlabels, yticks=yticks, minor_label_font_size=Hmatrix_font_size, vsize=Hdendrogram_vsize, hsize=Hdendrogram_hsize, color=dendrogram_color, background_color=background_color, quiet=quiet)
				catch errmsg
					!veryquiet && println(errmsg)
					@warn("H matrix dendrogram ploting failed!")
				end
			end
			if createbiplots
				@info("Biploting H matrix ...")
				P = permutedims(Ha) ./ maximum(Ha)
				Pm = vec(sum(P; dims=2))
				biplotlabels = string.(copy(Hnames))
				ilabel = Pm .< cutoff_label
				biplotlabels[ilabel] .= ""
				NMFk.biplots(P, biplotlabels, collect(1:k); smartplotlabel=true, filename="$figuredir/$(Hcasefilename)-$(k)-biplots-original.$(biplotformat)", background_color=background_color, types=chnew, plotlabel=Hbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				NMFk.biplots(P[cs,signalmap], biplotlabels[cs], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Hcasefilename)-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, types=chnew[cs], plotlabel=Hbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				length(Htypes) > 0 && NMFk.biplots(P[cs,signalmap], biplotlabels[cs], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Hcasefilename)-$(k)-biplots-type.$(biplotformat)", background_color=background_color, colors=Hcolors[cs], plotlabel=Hbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
			end
		end

		@info("$(uppercasefirst(Wcasefilename)) (signals=$k)")
		DelimitedFiles.writedlm("$resultdir/Wmatrix-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k))]; Wnames Wa], ',')
		if cutoff > 0
			ia = (Wa ./ maximum(Wa; dims=1)) .> cutoff
			for i in 1:k
				@info("Signal $i (max-normalized elements > $cutoff)")
				display(Wnames[ia[:,i]])
			end
		end

		if clusterW
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(cw, [i]) .== true
				@info("Signal $i (S$(wsignalmap[j])) Count: $(sum(ii))")
			end
			wsignalremap = indexin(signalmap, wsignalmap)
			cassgined = zeros(Int64, length(Wnames))
			cwnew = Vector{eltype(cw)}(undef, length(cw))
			cwnew .= ' '
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(cw, [clusterlabels[wsignalremap[j]]]) .== true
				cwnew[ii] .= i
				cassgined[ii] .+= 1
				@info("Signal $(clusterlabels[wsignalremap[j]]) -> $(i) Count: $(sum(ii))")
			end
			Wclusters[ki] = cwnew
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
				write(io, "Signal $i\n")
				ii = indexin(cwnew, [i]) .== true
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
			# snew2 = copy(snew)
			# for i = 1:k
			# 	snew2[snew .== "S$(i)"] .= "S$(ws[i])"
			# end
			dumpcsv = true
			if plotmap && length(lon) == length(cwnew)
				if isnothing(hover)
					hover = Wnames
				end
				if plotmap_scope == :well
					NMFk.plot_wells("$(Wcasefilename)-$(k)-map.$(map_format)", lon, lat, cwnew; figuredir=figuredir, hover=hover, title="Signals: $k")
				elseif plotmap_scope == :mapbox
					NMFk.mapbox(lon, lat, cwnew; filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, title="Signals: $k", map_kw...)
					NMFk.mapbox(lon, lat, Wm[:,signalmap], clusterlabels; filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map.$(map_format)"), text=hover, showlabels=true, map_kw...)
				else
					NMFk.plotmap(lon, lat, cwnew; filename=joinpath(figuredir, "$(Wcasefilename)-$(k)-map.$(map_format)"), title="Signals: $k", scope=string(plotmap_scope), map_kw...)
				end
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(clusterlabels) "Signal"]; Wnames lon lat Wm[:,signalmap] cwnew], ',')
				dumpcsv = false
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" permutedims(clusterlabels) "Signal"]; Wnames Wm[:,signalmap] cwnew], ',')
			end
			if !isnothing(lon) && (length(lon) != length(chnew)) && (length(lon) != length(cwnew))
				@warn("Length of lat/lon coordinates ($(length(lon))) does not match the number of either W matrix rows ($(length(cwnew))) or H matrix columns ($(length(chnew)))!")
			end
			cs = sortperm(cwnew)
			if createplots
				if length(cw) >= 100
					@warn("W ($(Wcasefilename)) matrix has too many attributes to plot as a matrix!")
				else
					xticks = ["S$i" for i=1:k]
					yticks = ["$(Wnames[i]) $(cw[i])" for i=eachindex(cw)]
					NMFk.plotmatrix(Wm; filename="$figuredir/$(Wcasefilename)-$(k)-original.$(plotmatrixformat)", xticks=xticks, yticks=yticks, colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color)
					# sorted by Wa magnitude
					# ws = sortperm(vec(sum(Wa; dims=1)); rev=true)
					# NMFk.plotmatrix(Wm[:,ws]; filename="$figuredir/$(Wcasefilename)-$(k)-original-sorted.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=eachindex(cw)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					cws = sortperm(cw)
					yticks = ["$(Wnames[cws][i]) $(cw[cws][i])" for i=eachindex(cw)]
					NMFk.plotmatrix(Wm[cws,:]; filename="$figuredir/$(Wcasefilename)-$(k)-original-sorted.$(plotmatrixformat)", xticks=xticks, yticks=yticks, colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
					yticks = ["$(Wnames[i]) $(cwnew[i])" for i=eachindex(cwnew)]
					NMFk.plotmatrix(Wm[:,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
					if length(Wtypes) > 0
						yticks = ["$(Wnametypes[i]) $(cwnew[i])" for i=eachindex(cwnew)]
						NMFk.plotmatrix(Wm[:,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-types.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=true, minor_label_font_size=Hmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, quiet=quiet)
					end
					yticks = ["$(Wnames[cs][i]) $(cwnew[cs][i])" for i=eachindex(cwnew)]
					NMFk.plotmatrix(Wm[cs,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize, background_color=background_color, quiet=quiet)
					# NMFk.plotmatrix(Wa./sum(Wa; dims=1); filename="$figuredir/$(Wcasefilename)-$(k)-sum.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=eachindex(cols)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					# NMFk.plotmatrix((Wa./sum(Wa; dims=1))[cs,:]; filename="$figuredir/$(Wcasefilename)-$(k)-sum2.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[cs][i]) $(cw[cs][i])" for i=eachindex(cols)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
					# NMFk.plotmatrix((Wa ./ sum(Wa; dims=1))[cs,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-labeled-sorted-sumrows.$(plotmatrixformat)", xticks=clusterlabels, yticks=["$(Wnames[cs][i]) $(cwnew[cs][i])" for i=eachindex(cwnew)], colorkey=true, minor_label_font_size=Wmatrix_font_size, vsize=Wmatrix_vsize, hsize=Wmatrix_hsize)
				end
				if plottimeseries == :W || plottimeseries == :WH
					Mads.plotseries(Wa ./ maximum(Wa), "$figuredir/$(Wcasefilename)-$(k)-timeseries.$(plotseriesformat)"; xaxis=Wnames, xmin=minimum(Wnames), xmax=maximum(Wnames), vsize=Wtimeseries_vsize, hsize=Wtimeseries_hsize)
				end
			end
			if (createdendrogramsonly || createplots) && length(cw) < 100
				@info("Dendrogram ploting ...")
				try
					yticks = ["$(Wnames[cs][i]) $(cwnew[cs][i])" for i=eachindex(cwnew)]
					NMFk.plotdendrogram(Wm[cs,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted-dendrogram.$(plotmatrixformat)", metricheat=nothing, xticks=clusterlabels, yticks=yticks, minor_label_font_size=Wmatrix_font_size, vsize=Wdendrogram_vsize, hsize=Wdendrogram_hsize, color=dendrogram_color, background_color=background_color, quiet=quiet)
				catch errmsg
					!veryquiet && println(errmsg)
					@warn("W matrix dendrogram ploting failed!")
				end
			end
			if createbiplots
				@info("Biploting W matrix ...")
				P = Wa ./ maximum(Wa)
				Pm = vec(sum(P; dims=2))
				biplotlabels = string.(copy(Wnames))
				ilabel = Pm .< cutoff_label
				biplotlabels[ilabel] .= ""
				NMFk.biplots(P, biplotlabels, collect(1:k); smartplotlabel=true, filename="$figuredir/$(Wcasefilename)-$(k)-biplots-original.$(biplotformat)", background_color=background_color, types=cwnew, plotlabel=Wbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				NMFk.biplots(P[cs,signalmap], biplotlabels[cs], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Wcasefilename)-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, types=cwnew[cs], plotlabel=Wbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size, quiet=quiet)
				length(Wtypes) > 0 && NMFk.biplots(P[cs,signalmap], biplotlabels[cs], clusterlabels; smartplotlabel=true, filename="$figuredir/$(Wcasefilename)-$(k)-biplots-type.$(biplotformat)", background_color=background_color, colors=Wcolors[cs], plotlabel=Wbiplotlabel, sortmag=sortmag, plotmethod=plotmethod, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
			end
			if createbiplots
				@info("Biploting combined W and H matrices ...")
				M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))]
				Mm = vec(sum(M; dims=2))
				Wnames_label = string.(copy(Wnames))
				Hnames_label = string.(copy(Hnames))
				ilabel = Mm .< cutoff_label
				if biplotlabel == :W
					biplotlabels = [Wnames_label; fill("", length(Hnames))]
					biplotlabels[ilabel] .= ""
					biplotlabelflag = true
				elseif biplotlabel == :WH
					biplotlabels = [Wnames_label; Hnames_label]
					biplotlabels[ilabel] .= ""
					biplotlabelflag = true
				elseif biplotlabel == :H
					biplotlabels = [fill("", length(Wnames)); Hnames_label]
					biplotlabels[ilabel] .= ""
					biplotlabelflag = true
				elseif biplotlabel == :none
					biplotlabels = [fill("", length(Wnames)); fill("", length(Hnames))]
					biplotlabelflag = false
				end
				Wbiplottypecolors = length(Wtypes) > 0 ? Wcolors : set_typecolors(cwnew, Wcolors)
				Hbiplottypecolors = length(Htypes) > 0 ? Hcolors : set_typecolors(chnew, Hcolors)
				if biplotcolor == :W
					biplotcolors = [Wbiplottypecolors; fill("gray", length(Hnames))]
				elseif biplotcolor == :WH
					Hbiplottypecolors = length(Htypes) > 0 ? Hcolors : set_typecolors(chnew, Hcolors[k+1:end])
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