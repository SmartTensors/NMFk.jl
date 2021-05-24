import DelimitedFiles
import PlotlyJS
import Mads

function signal_importance(krange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector)
	signal_order = Array{Array{Int64}}(undef, maximum(krange))
	for k = 1:maximum(krange)
		signal_order[k] = Array{Int64}(undef, 0)
	end
	for k in krange
		@info("Number of signals: $k")
		signal_order[k] = signal_importance(W[k], H[k])
	end
	return signal_order
end

function signal_importance(W::AbstractMatrix, H::AbstractMatrix)
	k = size(W, 2)
	@assert k == size(H, 1)
	signal_sum = Array{eltype(W)}(undef, k)
	for i = 1:k
		signal_sum[i] = sum(W[:,i:i] * H[i:i,:])
	end
	signal_order = sortperm(signal_sum; rev=true)
	println("Signal importance (high->low): $signal_order")
	return signal_order
end

function plot_signal_selecton(nkrange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, fitquality::AbstractVector, robustness::AbstractVector, X::AbstractMatrix, W::AbstractVector, H::AbstractVector; figuredir::AbstractString=".", casefilename::AbstractString="signal_selection", title::AbstractString="", xtitle::AbstractString="Number of signals", ytitle::AbstractString="Normalized metrics", plotformat::AbstractString="png", normalize_robustness::Bool=true, kw...)
	r = normalize_robustness ? robustness[nkrange] ./ maximumnan(robustness[nkrange]) : robustness[nkrange]
	r2 = similar(robustness)
	for k in nkrange
		r2[k] = NMFk.r2(X, W[k] * H[k])
		NMFk.plotscatter(X, W[k] * H[k]; title="Number of Signals = $k R2 = $(r2[k])", ymax=1, xmax=1)
	end
	Mads.plotseries([fitquality[nkrange] ./ maximumnan(fitquality[nkrange]) r r2[nkrange]], "$(figuredir)/$(casefilename).$(plotformat)"; title=title, ymin=0, xaxis=nkrange, xmin=nkrange[1], xtitle=xtitle, ytitle=ytitle, names=["Fit", "Robustness", "R2"], kw...)
end

function plot_signal_selecton(nkrange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, fitquality::AbstractVector, robustness::AbstractVector; figuredir::AbstractString=".", casefilename::AbstractString="signal_selection", title::AbstractString="", xtitle::AbstractString="Number of signals", ytitle::AbstractString="Normalized metrics", plotformat::AbstractString="png", normalize_robustness::Bool=true, kw...)
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
	for i = 1:size(X, 1)
		@info "Signal $i"
		is = order(Xm[:,i])
		ivl = filter_vals(Xm[:,i][is])
		inm = filter_names(Xnames[is][1:ivl])
		display([Xnames[is] Xm[:,i][is]][1:ivl,:][inm,:])
	end
end

function clusterresults(W::AbstractMatrix{T}, H::AbstractMatrix{T}, aw...; kw...) where {T <: Number}
	k = size(W, 2)
	@assert size(H, 1) == k
	Wa = Array{Array{T, 2}}(undef, k)
	Ha = Array{Array{T, 2}}(undef, k)
	Wa[k] = W
	Ha[k] = H
	NMFk.clusterresults(k, Wa, Ha, aw...; kw...)
end

function clusterresults(nkrange::AbstractRange{Int}, nruns::Integer, Wnames::AbstractVector, Hnames::AbstractVector; cutoff::Number=0.5, kw...)
	NMFk.clusterresults(NMFk.getks(nkrange, silhouette[nkrange], cutoff), nkrange, nruns, Wnames, Hnames; kw...)
end

function clusterresults(krange::Union{AbstractVector{Int64},Integer}, nkrange::AbstractRange{Int}, nruns::Integer, Wnames::AbstractVector, Hnames::AbstractVector; resultdir::AbstractString=".", casefilename::AbstractString="nmfk", keyword::AbstractString="", kw...)
	W, H, fit, silhouette, aic, kopt = NMFk.load(nkrange, nruns; resultdir=resultdir, casefilename=casefilename)
	suffix = "$(casefilename)-$(nruns)"
	NMFk.clusterresults(krange, W, H, Wnames, Hnames; resultdir="results-$(suffix)", figuredir="figures-$(suffix)", kw...)
end

"""
cutoff::Number = .9, cutoff_s::Number = 0.95
"""
function clusterresults(krange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector, Wnames::AbstractVector, Hnames::AbstractVector; ordersignal::Symbol=:importance, clusterW::Bool=true, clusterH::Bool=true, loadassignements::Bool=true, Wsize::Integer=0, Hsize::Integer=0, Wmap::AbstractVector=[], Hmap::AbstractVector=[], Worder::AbstractVector=collect(1:length(Wnames)), Horder::AbstractVector=collect(1:length(Hnames)), lon=nothing, lat=nothing, hover=nothing, resultdir::AbstractString=".", figuredir::AbstractString=resultdir, Wcasefilename::AbstractString="attributes", Hcasefilename::AbstractString="locations", Htypes::AbstractVector=[], Wtypes::AbstractVector=[], Hcolors=NMFk.colors, Wcolors=NMFk.colors, background_color="black", createplots::Bool=true, createbiplots::Bool=createplots, Wbiplotlabel::Bool=!(length(Wnames) > 20), Hbiplotlabel::Bool=!(length(Hnames) > 20), plottimeseries::Symbol=:none, biplotlabel::Symbol=:none, biplotcolor::Symbol=:WH, cutoff::Number=0, cutoff_s::Number=0, Wmatrix_font_size=10Gadfly.pt, Hmatrix_font_size=10Gadfly.pt, plotmatrixformat="png", biplotformat="pdf", plotseriesformat="png", sortmag::Bool=false, point_size_nolabel=3Gadfly.pt, point_size_label=3Gadfly.pt, biplotseparate::Bool=false, biplot_point_label_font_size=12Gadfly.pt)
	@assert length(Wnames) == length(Worder)
	@assert length(Hnames) == length(Horder)
	@assert any(Worder .== nothing) == false
	@assert any(Horder .== nothing) == false
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
	Wnamesmaxlength = max(length.(Wnames)...)
	Wnames = Wnames[Worder]
	Hnames = Hnames[Horder]
	if lon !== nothing && lat !== nothing
		@assert length(lon) == length(lat)
		plotmap = 0
	end
	Wclusters = Vector{Vector{Char}}(undef, length(krange))
	Hclusters = Vector{Vector{Char}}(undef, length(krange))
	Sorder = Vector{Vector{Int64}}(undef, length(krange))
	for (ki, k) in enumerate(krange)
		@info("Number of signals: $k")

		isignalmap = signal_importance(W[k], H[k])

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
				@info "Signal $i (max-normalized elements > $cutoff)"
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
				mu = unique(Ws)
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

		if clusterH
			ch = NMFk.labelassignements(NMFk.robustkmeans(Ha, k; resultdir=resultdir, casefilename="Hmatrix", load=loadassignements, save=true)[1].assignments)
			clusterlabels = sort(unique(ch))
			hsignalmap = NMFk.getsignalassignments(Ha, ch; clusterlabels=clusterlabels, dims=2)
		end

		if clusterW
			cw = NMFk.labelassignements(NMFk.robustkmeans(permutedims(Wa), k; resultdir=resultdir, casefilename="Wmatrix", load=loadassignements, save=true)[1].assignments)
			if clusterH
				@assert clusterlabels == sort(unique(cw))
			else
				clusterlabels = sort(unique(cw))
			end
			wsignalmap = NMFk.getsignalassignments(Wa, cw; clusterlabels=clusterlabels, dims=1)
		end

		if ordersignal == :importance
			signalmap = isignalmap
		elseif ordersignal == :Hcount && clusterH
			signalmap = hsignalmap
		elseif ordersignal == :Wcount && clusterW
			signalmap = wsignalmap
		else
			@warn "Unknown signal order requested $(ordersignal); Signal importance will be used!"
			signalmap = isignalmap
		end
		Sorder[ki] = signalmap

		if clusterH
			signalhmap = indexin(signalmap, hsignalmap)
			cassgined = zeros(Int64, length(Hnames))
			chnew = Vector{eltype(ch)}(undef, length(ch))
			chnew .= ' '
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(ch, [clusterlabels[signalhmap[j]]]) .== true
				chnew[ii] .= i
				cassgined[ii] .+= 1
				@info "Signal $(clusterlabels[signalhmap[j]]) -> $(i) Count: $(sum(ii))"
			end
			Hclusters[ki] = chnew
			if any(cassgined .== 0)
				@warn "$(uppercasefirst(Hcasefilename)) not assigned to any cluster:"
				display(Hnames[cassgined .== 0])
				@error "Something is wrong!"
			end
			if any(cassgined .> 1)
				@warn "$(uppercasefirst(Hcasefilename)) assigned to more than cluster:"
				display([Hnames[cassgined .> 1] cassgined[cassgined .> 1]])
				@error "Something is wrong!"
			end
			clustermap = Vector{Char}(undef, k)
			clustermap .= ' '
			io = open("$resultdir/$(Hcasefilename)-$(k)-groups.txt", "w")
			for (j, i) in enumerate(clusterlabels)
				@info "Signal $i (S$(signalmap[j])) (k-means clustering)"
				write(io, "Signal $i (S$(signalmap[j]))\n")
				ii = indexin(chnew, [i]) .== true
				is = sortperm(Hm[ii,signalmap[j]]; rev=true)
				d = [Hnames[ii] Hm[ii,signalmap[j]]][is,:]
				display(d)
				for i = 1:size(d, 1)
					write(io, "$(rpad(d[i,1], Hnamesmaxlength))\t$(round(d[i,2]; sigdigits=3))\n")
				end
				write(io, '\n')
				clustermap[signalmap[j]] = i
			end
			close(io)
			@assert signalmap == sortperm(clustermap)
			@assert clustermap[signalmap] == clusterlabels
			dumpcsv = true
			if lon !== nothing && lat !== nothing
				if length(lon) != length(chnew)
					plotmap = 1
				else
					NMFk.plot_wells("$(Hcasefilename)-$(k)-map.html", lon, lat, chnew; figuredir=figuredir, hover=hover, title="Signals: $k")
					lonlat = [lon lat]
					DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(clusterlabels) "Signal"]; Hnames lonlat Hm[:,signalmap] chnew], ',')
					dumpcsv = false
				end
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" permutedims(clusterlabels) "Signal"]; Hnames Hm[:,signalmap] chnew], ',')
			end
			cs = sortperm(chnew)
			if createplots
				xticks = ["S$i" for i=1:k]
				yticks = ["$(Hnames[i]) $(chnew[i])" for i=1:length(chnew)]
				NMFk.plotmatrix(Hm; filename="$figuredir/$(Hcasefilename)-$(k)-original.$(plotmatrixformat)", xticks=xticks, yticks=yticks, colorkey=false, minor_label_font_size=Hmatrix_font_size)
				NMFk.plotmatrix(Hm[:,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=false, quiet=false, minor_label_font_size=Hmatrix_font_size)
				if length(Htypes) > 0
					yticks = ["$(Hnametypes[i]) $(chnew[i])" for i=1:length(chnew)]
					NMFk.plotmatrix(Hm[:,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-types.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=false, minor_label_font_size=Hmatrix_font_size)
				end
				yticks = ["$(Hnames[cs][i]) $(chnew[cs][i])" for i=1:length(chnew)]
				NMFk.plotmatrix(Hm[cs,signalmap]; filename="$figuredir/$(Hcasefilename)-$(k)-labeled-sorted.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=false, quiet=false, minor_label_font_size=Hmatrix_font_size)
				try
					display(NMFk.plotdendrogram(Hm[cs,signalmap]; metricheat=nothing, xticks=clusterlabels, yticks=yticks, minor_label_font_size=Hmatrix_font_size))
				catch errmsg
					println(errmsg)
					@warn("Dendogram ploting failed!")
				end
				if plottimeseries == :H || plottimeseries == :WH
					Mads.plotseries(Hm, "$figuredir/$(Hcasefilename)-$(k)-timeseries.$(plotseriesformat)"; xaxis=Hnames)
				end
			end
			if createbiplots
				NMFk.biplots(permutedims(Ha) ./ maximum(Ha), Hnames, collect(1:k); filename="$figuredir/$(Hcasefilename)-$(k)-biplots-original.$(biplotformat)", background_color=background_color, types=chnew, plotlabel=Hbiplotlabel, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
				NMFk.biplots(permutedims(Ha)[cs,signalmap] ./ maximum(Ha), Hnames[cs], clusterlabels; filename="$figuredir/$(Hcasefilename)-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, types=chnew[cs], plotlabel=Hbiplotlabel, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
				length(Htypes) > 0 && NMFk.biplots(permutedims(Ha)[cs,signalmap]./ maximum(Ha), Hnames[cs], clusterlabels; filename="$figuredir/$(Hcasefilename)-$(k)-biplots-type.$(biplotformat)", background_color=background_color, colors=Hcolors[cs], plotlabel=Hbiplotlabel, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
			end
		end

		@info("$(uppercasefirst(Wcasefilename)) (signals=$k)")
		DelimitedFiles.writedlm("$resultdir/Wmatrix-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k))]; Wnames Wa], ',')
		if cutoff > 0
			ia = (Wa ./ maximum(Wa; dims=1)) .> cutoff
			for i in 1:k
				@info "Signal $i (max-normalized elements > $cutoff)"
				display(Wnames[ia[:,i]])
			end
		end

		if clusterW
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(cw, [i]) .== true
				@info "Signal $i (S$(wsignalmap[j])) Count: $(sum(ii))"
			end
			signalwmap = indexin(signalmap, wsignalmap)
			cassgined = zeros(Int64, length(Wnames))
			cwnew = Vector{eltype(cw)}(undef, length(cw))
			cwnew .= ' '
			for (j, i) in enumerate(clusterlabels)
				ii = indexin(cw, [clusterlabels[signalwmap[j]]]) .== true
				cwnew[ii] .= i
				cassgined[ii] .+= 1
				@info "Signal $(clusterlabels[signalwmap[j]]) -> $(i) Count: $(sum(ii))"
			end
			Wclusters[ki] = cwnew
			if any(cassgined .== 0)
				@warn "$(uppercasefirst(Wcasefilename)) not assigned to any cluster:"
				display(Wnames[cassgined .== 0])
				@error "Something is wrong!"
			end
			if any(cassgined .> 1)
				@warn "$(uppercasefirst(Wcasefilename)) assigned to more than cluster:"
				display([Wnames[cassgined .> 1] cassgined[cassgined .> 1]])
				@error "Something is wrong!"
			end
			io = open("$resultdir/$(Wcasefilename)-$(k)-groups.txt", "w")
			for (j, i) in enumerate(clusterlabels)
				@info "Signal $i (remapped k-means clustering)"
				write(io, "Signal $i\n")
				ii = indexin(cwnew, [i]) .== true
				is = sortperm(Wm[ii,signalmap[j]]; rev=true)
				d = [Wnames[ii] Wm[ii,signalmap[j]]][is,:]
				display(d)
				for i = 1:size(d, 1)
					write(io, "$(rpad(d[i,1], Wnamesmaxlength))\t$(round(d[i,2]; sigdigits=3))\n")
				end
				write(io, '\n')
			end
			close(io)
			# snew2 = copy(snew)
			# for i = 1:k
			# 	snew2[snew .== "S$(i)"] .= "S$(ws[i])"
			# end
			dumpcsv = true
			if lon !== nothing && lat !== nothing
				if length(lon) != length(cwnew)
					(plotmap == 1) && @warn("Coordinate data does not match the number of either W matrix rows or H matrix columns!")
				else
					NMFk.plot_wells("$(Wcasefilename)-$(k)-map.html", lon, lat, cwnew; figuredir=figuredir, hover=hover, title="Signals: $k")
					lonlat = [lon lat]
					DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(clusterlabels) "Signal"]; Wnames lonlat Wm[:,signalmap] cwnew], ',')
					dumpcsv = false
				end
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" permutedims(clusterlabels) "Signal"]; Wnames Wm[:,signalmap] cwnew], ',')
			end
			cs = sortperm(cwnew)
			if createplots
				xticks = ["S$i" for i=1:k]
				yticks = ["$(Wnames[i]) $(cw[i])" for i=1:length(cw)]
				NMFk.plotmatrix(Wm; filename="$figuredir/$(Wcasefilename)-$(k)-original.$(plotmatrixformat)", xticks=xticks, yticks=yticks, colorkey=false, minor_label_font_size=Wmatrix_font_size)
				# sorted by Wa magnitude
				# ws = sortperm(vec(sum(Wa; dims=1)); rev=true)
				# NMFk.plotmatrix(Wm[:,ws]; filename="$figuredir/$(Wcasefilename)-$(k)-original-sorted.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				cws = sortperm(cw)
				yticks = ["$(Wnames[cws][i]) $(cw[cws][i])" for i=1:length(cw)]
				NMFk.plotmatrix(Wm[cws,:]; filename="$figuredir/$(Wcasefilename)-$(k)-original-sorted.$(plotmatrixformat)", xticks=xticks, yticks=yticks, colorkey=false, minor_label_font_size=Wmatrix_font_size)
				yticks = ["$(Wnames[i]) $(cwnew[i])" for i=1:length(cwnew)]
				NMFk.plotmatrix(Wm[:,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=false, quiet=false, minor_label_font_size=Wmatrix_font_size)
				if length(Wtypes) > 0
					yticks = ["$(Wnametypes[i]) $(cwnew[i])" for i=1:length(cwnew)]
					NMFk.plotmatrix(Wm[:,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-types.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=false, minor_label_font_size=Hmatrix_font_size)
				end
				yticks = ["$(Wnames[cs][i]) $(cwnew[cs][i])" for i=1:length(cwnew)]
				NMFk.plotmatrix(Wm[cs,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-remappped-sorted.$(plotmatrixformat)", xticks=clusterlabels, yticks=yticks, colorkey=false, quiet=false, minor_label_font_size=Wmatrix_font_size)
				try
					display(NMFk.plotdendrogram(Wm[cs,signalmap]; metricheat=nothing, xticks=clusterlabels, yticks=yticks, minor_label_font_size=Wmatrix_font_size))
				catch errmsg
					println(errmsg)
					@warn("Dendogram ploting failed!")
				end
				# NMFk.plotmatrix(Wa./sum(Wa; dims=1); filename="$figuredir/$(Wcasefilename)-$(k)-sum.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=1:length(cols)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				# NMFk.plotmatrix((Wa./sum(Wa; dims=1))[cs,:]; filename="$figuredir/$(Wcasefilename)-$(k)-sum2.$(plotmatrixformat)", xticks=["S$i" for i=1:k], yticks=["$(Wnames[cs][i]) $(cw[cs][i])" for i=1:length(cols)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				# NMFk.plotmatrix((Wa ./ sum(Wa; dims=1))[cs,signalmap]; filename="$figuredir/$(Wcasefilename)-$(k)-labeled-sorted-sumrows.$(plotmatrixformat)", xticks=clusterlabels, yticks=["$(Wnames[cs][i]) $(cwnew[cs][i])" for i=1:length(cwnew)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				if plottimeseries == :W || plottimeseries == :WH
					Mads.plotseries(Wa ./ maximum(Wa), "$figuredir/$(Wcasefilename)-$(k)-timeseries.$(plotseriesformat)"; xaxis=Wnames)
				end
			end
			if createbiplots
				NMFk.biplots(Wa ./ maximum(Wa), Wnames, collect(1:k); filename="$figuredir/$(Wcasefilename)-$(k)-biplots-original.$(biplotformat)", background_color=background_color, types=cwnew, plotlabel=Wbiplotlabel, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
				NMFk.biplots(Wa[cs,signalmap] ./ maximum(Wa), Wnames[cs], clusterlabels; filename="$figuredir/$(Wcasefilename)-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, types=cwnew[cs], plotlabel=Wbiplotlabel, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
				length(Wtypes) > 0 && NMFk.biplots(Wa[cs,signalmap] ./ maximum(Wa), Wnames[cs], clusterlabels; filename="$figuredir/$(Wcasefilename)-$(k)-biplots-type.$(biplotformat)", background_color=background_color, colors=Wcolors[cs], plotlabel=Wbiplotlabel, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
			end
			if createbiplots
				if biplotlabel == :W
					biplotlabels = [Wnames; fill("", length(Hnames))]
					biplotlabelflag = true
				elseif biplotlabel == :WH
					biplotlabels = [Wnames; Hnames]
					biplotlabelflag = true
				elseif biplotlabel == :H
					biplotlabels = [fill("", length(Wnames)); Hnames]
					biplotlabelflag = true
				elseif biplotlabel == :none
					biplotlabels = [fill("", length(Wnames)); fill("", length(Hnames))]
					biplotlabelflag = false
				end
				Wbiplottypecolors = length(Wtypes) > 0 ? Wcolors : typecolors(cwnew, Wcolors)
				Hbiplottypecolors = length(Htypes) > 0 ? Hcolors : typecolors(chnew, Hcolors)
				if biplotcolor == :W
					biplotcolors = [Wbiplottypecolors; fill("gray", length(Hnames))]
				elseif biplotcolor == :WH
					Hbiplottypecolors = length(Htypes) > 0 ? Hcolors : typecolors(chnew, Hcolors[k+1:end])
					biplotcolors = [Wbiplottypecolors; Hbiplottypecolors]
				elseif biplotcolor == :H
					biplotcolors = [fill("gray", length(Wnames)); Hbiplottypecolors]
				elseif biplotcolor == :none
					biplotcolors = [fill("blue", length(Wnames)); fill("red", length(Hnames))]
				end
				M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))]
				NMFk.biplots(M, biplotlabels, collect(1:k); filename="$figuredir/all-$(k)-biplots-original.$(biplotformat)", background_color=background_color, typecolors=biplotcolors, plotlabel=biplotlabelflag, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
				if biplotcolor == :W
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,signalmap]
				elseif biplotcolor == :WH
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,signalmap]
				elseif biplotcolor == :H
					M = [permutedims(Ha ./ maximum(Ha)); Wa ./ maximum(Wa)][:,signalmap]
				elseif biplotcolor == :none
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,signalmap]
				end
				NMFk.biplots(M, biplotlabels, clusterlabels; filename="$figuredir/all-$(k)-biplots-labeled.$(biplotformat)", background_color=background_color, typecolors=biplotcolors, plotlabel=biplotlabelflag, sortmag=sortmag, point_size_nolabel=point_size_nolabel, point_size_label=point_size_label, separate=biplotseparate, point_label_font_size=biplot_point_label_font_size)
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
				table = hcat(table, map(i->attributesl[i], map(i->o[2][i][1], 1:length(Hnames))))
				table2 = hcat(table2, map(i->attributesl[Xekm[:,i]], 1:length(Hnames)))
				table3 = hcat(table3, map(i->sum(Xekm[:,i]), 1:length(Hnames)))
			end
			if lon !== nothing && lat !== nothing
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_max.csv", [lonlat table], ',')
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_$(cutoff_s).csv", [lonlat table2], ';')
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k)-table_count_$(cutoff_s).csv", [lonlat table3], ',')
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
				table = hcat(table, map(i->Hnames[i], map(i->o[2][i][2], 1:length(attributesl))))
				table2 = hcat(table2, map(i->Hnames[Xekm[i,:]], 1:length(attributesl)))
				table3 = hcat(table3, map(i->sum(Xekm[i,:]), 1:length(attributesl)))
			end
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k)-table_max.csv", table, ',')
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k)-table_$(cutoff_s).csv", table2, ';')
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k)-table_count_$(cutoff_s).csv", table3, ',')
		end
	end
	return Sorder, Wclusters, Hclusters
end

function signalorder(X::AbstractArray, dim=1)
	v = Vector{Int64}(undef, size(X, dim))
	for i = 1:size(X, dim)
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(X))
		v[i] = findmax(X[nt...])[2]
	end
	sortperm(v), v
end

function signalorder(W::AbstractMatrix, H::AbstractMatrix; resultdir::AbstractString=".", loadassignements::Bool=true, Wclusterlabelcasefilename::AbstractString="Wmatrix", Hclusterlabelcasefilename::AbstractString="Hmatrix")
	k = size(H, 1)
	Hclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(H, k; resultdir=resultdir, casefilename=Hclusterlabelcasefilename, load=loadassignements, save=true)[1].assignments)
	Hcs = sortperm(Hclusterlabels)
	clusterlabels = sort(unique(Hclusterlabels))
	Hsignalmap = NMFk.getsignalassignments(H, Hclusterlabels; clusterlabels=clusterlabels, dims=2)
	Hclustermap = Vector{Char}(undef, k)
	Hclustermap .= ' '
	Hsignals = Vector{String}(undef, length(Hclusterlabels))
	for (j, i) in enumerate(clusterlabels)
		Hclustermap[Hsignalmap[j]] = i
		Hsignals[Hclusterlabels .== i] .= "S$(Hsignalmap[j])"
	end
	Wclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(permutedims(W), k; resultdir=resultdir, casefilename=Wclusterlabelcasefilename, load=loadassignements, save=true)[1].assignments)
	@assert clusterlabels == sort(unique(Wclusterlabels))
	Wsignalmap = NMFk.getsignalassignments(W[:,Hsignalmap], Wclusterlabels; clusterlabels=clusterlabels, dims=1)
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

function getmissingattributes(X::AbstractMatrix, attributes::AbstractVector, locationclusters::AbstractVector; locationmatrix::Union{Nothing,AbstractMatrix}=nothing, attributematrix::Union{Nothing,AbstractMatrix}=nothing, dims::Integer=2, plothistogram::Bool=false, quiet::Bool=true)
	for (ic, c) in enumerate(unique(sort(locationclusters)))
		i = locationclusters .== c
		@info "Location cluster: $c"
		min, max, std, count = NMFk.datanalytics(X[i,:], attributes; dims=dims, plothistogram=plothistogram, quiet=quiet)
		@info "Missing attribute measurements:"
		if attributematrix === nothing
			display(attributes[count.==0])
		else
			p = attributematrix[ic,count.==0]
			is = sortperm(p; rev=true)
			display([attributes[count.==0] p][is,:])
		end
	end
end