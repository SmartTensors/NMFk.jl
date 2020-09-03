import DelimitedFiles
import PlotlyJS
import Mads

function plot_feature_selecton(nkrange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, fitquality::AbstractVector, robustness::AbstractVector; figuredir::AbstractString=".", casefilename::AbstractString="feature_selection", title::AbstractString="")
	Mads.plotseries([fitquality[nkrange] ./ maximumnan(fitquality[nkrange]) robustness[nkrange]], "$(figuredir)/$(casefilename).png"; title=title, ymin=0, xaxis=nkrange, xmin=nkrange[1], names=["Fit", "Robustness"])
end

function showsignatures(X::AbstractMatrix, Xnames::AbstractVector; Xmap::AbstractVector=[], order::Function=i->sortperm(i; rev=true), select::Function=v->findlast(i->i>0.95, v))
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
		@info "Signature $i"
		is = order(Xm[:,i])
		il = select(Xm[:,i][is])
		display([Xnames[is] Xm[:,i][is]][1:il,:])
	end
end

function clusterresults(nkrange::Union{AbstractRange{Int},AbstractVector{Int64}}, W::AbstractVector, H::AbstractVector, robustness::AbstractVector, Hnames::AbstractVector, Wnames::AbstractVector; kw...)
	krange = NMFk.getks(nkrange, robustness[nkrange])
	if length(krange) == 0
		@warn("Optimal number of signals cannot be determined!")
	else
		@warn("Optimal number of signals: $(krange)")
		if length(Wnames) == size(W[krange[1]], 1)
			clusterresults(krange, W, H, Wnames, Hnames; kw...)
		else
			clusterresults(krange, W, H, Hnames, Wnames; kw...)
		end
	end
end

"""
cutoff::Number = .9, cutoff_s::Number = 0.95
"""
function clusterresults(krange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector, Wnames::AbstractVector, Hnames::AbstractVector; clusterattributes::Bool=true, loadassignements::Bool=true, Wsize::Integer=0, Hsize::Integer=0, Wmap::AbstractVector=[], Hmap::AbstractVector=[], lon=nothing, lat=nothing, hover=nothing, resultdir::AbstractString=".", figuredir::AbstractString=resultdir, Wcasefilename::AbstractString="attributes", Hcasefilename::AbstractString="locations", Htypes::AbstractVector=[], Wtypes::AbstractVector=[], Hcolors=NMFk.colors, Wcolors=NMFk.colors, background_color="black", createplots::Bool=true, Hplotlabel::Bool=true, Wplotlabel::Bool=true, plottimeseries::Symbol=:none, biplotlabel::Symbol=:W, biplotcolor::Symbol=:W, cutoff::Number=0, cutoff_s::Number=0, Wmatrix_font_size=10Gadfly.pt, Hmatrix_font_size=10Gadfly.pt)
	if length(Htypes) > 0
		if Hcolors == NMFk.colors
			Hcolors = Vector{String}(undef, length(Htypes))
			for (j, t) in enumerate(unique(Htypes))
				Hcolors[Htypes .== t] .= NMFk.colors[j]
			end
		end
		Hnametypes = Hnames .* " " .* String.(Htypes)
	else
		Hnametypes = Hnames
	end
	if length(Wtypes) > 0
		if Wcolors == NMFk.colors
			Wcolors = Vector{String}(undef, length(Wtypes))
			for (j, t) in enumerate(unique(Wtypes))
				Wcolors[attributetype .== t] .= NMFk.colors[j]
			end
		end
		Wnametypes = Wnames .* " " .* String.(Wtypes)
	else
		Wnametypes = Wnames
	end
	if lon != nothing && lat != nothing
		@assert length(lon) == length(lat)
		plotmap = 0
	end
	for k in krange
		@info("Number of signals: $k")

		@info("$(uppercasefirst(Hcasefilename)) (signals=$k)")
		recursivemkdir(resultdir; filename=false)

		if Hsize > 1
			na = convert(Int64, size(H[k], 2) / Hsize)
			Wa = Matrix{eltype(H[k])}(undef, size(H[k], 1), na)
			@assert length(Hnames) == na
			i1 = 1
			i2 = Hsize
			for i = 1:na
				Ha[:,i] = sum(H[k][:,i1:i2]; dims=2)
				i1 += Hsize
				i2 += Hsize
			end
		elseif length(Hmap) > 0
			@assert length(Hmap) == size(H[k], 2)
			mu = unique(Hmap)
			na = length(mu)
			@assert length(Hnames) == na
			Ha = Matrix{eltype(H[k])}(undef, size(H[k], 1), na)
			for (i, m) in enumerate(mu)
				Ha[:,i] = sum(H[k][:, Hmap .== m]; dims=2)
			end
		else
			Ha = H[k]
			@assert length(Hnames) == size(Ha, 2)
		end

		DelimitedFiles.writedlm("$resultdir/Hmatrix-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k))]; Hnames permutedims(Ha)], ',')
		if cutoff > 0
			ia = (Ha ./ maximum(Ha; dims=2)) .> cutoff
			for i in 1:k
				@info "Signal $i (max-normalized elements > $cutoff)"
				display(Hnames[ia[i,:]])
			end
		end
		ch = NMFk.labelassignements(NMFk.robustkmeans(Ha, k; resultdir=resultdir, casefilename="Hmatrix", load=loadassignements, save=true)[1].assignments)
		cs = sortperm(ch)
		clusterlabels = sort(unique(ch))
		signalmap = NMFk.getsignalassignments(Ha, ch; clusterlabels=clusterlabels, dims=2)
		clustermap = Vector{Char}(undef, k)
		clustermap .= ' '
		Hm = permutedims(Ha ./ maximum(Ha; dims=2))
		io = open("$resultdir/$(Hcasefilename)-groups-$(k).txt", "w")
		for (j, i) in enumerate(clusterlabels)
			@info "Signal $i (S$(signalmap[j])) (k-means clustering)"
			write(io, "Signal $i (S$(signalmap[j])) (k-means clustering)\n")
			ii = indexin(ch, [i]) .== true
			is = sortperm(Hm[ii,signalmap[j]]; rev=true)
			d = [Hnames[ii] Hm[ii,signalmap[j]]][is,:]
			display(d)
			for l in d
				write(io, l)
				write(io, '\n')
			end
			write(io, '\n')
			clustermap[signalmap[j]] = i
		end
		close(io)
		is = signalmap
		@assert is == sortperm(clustermap)
		if createplots
			NMFk.plotmatrix(Hm; filename="$figuredir/$(Hcasefilename)-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Hnames[i]) $(ch[i])" for i=1:length(ch)], colorkey=false, minor_label_font_size=Hmatrix_font_size)
			NMFk.plotmatrix((Hm)[cs,is]; filename="$figuredir/$(Hcasefilename)-sorted-$(k).png", xticks=clustermap[is], yticks=["$(Hnames[cs][i]) $(ch[cs][i])" for i=1:length(ch)], colorkey=false, quiet=false, minor_label_font_size=Hmatrix_font_size)
			NMFk.plotmatrix(permutedims((Ha ./ sum(Ha; dims=2)))[cs,is]; filename="$figuredir/$(Hcasefilename)-sorted-sumrows-$(k).png", xticks=clustermap[is], yticks=["$(Hnametypes[cs][i]) $(ch[cs][i])" for i=1:length(ch)], colorkey=false, minor_label_font_size=Hmatrix_font_size)
			NMFk.biplots(permutedims(Ha) ./ maximum(Ha), Hnames, collect(1:k); filename="$figuredir/$(Hcasefilename)-biplots-$(k)-original.pdf", background_color=background_color, types=ch, plotlabel=Hplotlabel)
			NMFk.biplots(permutedims(Ha)[cs,is] ./ maximum(Ha), Hnames[cs], clustermap[is]; filename="$figuredir/$(Hcasefilename)-biplots-$(k).pdf", background_color=background_color, types=ch[cs], plotlabel=Hplotlabel)
			length(Htypes) > 0 && NMFk.biplots(permutedims(Ha)[cs,is]./ maximum(Ha), Hnames[cs], clustermap[is]; filename="$figuredir/$(Hcasefilename)-biplots-type-$(k).pdf", background_color=background_color, colors=Hcolors[cs], plotlabel=Hplotlabel)
			if plottimeseries == :H
				Mads.plotseries(Hm, "$figuredir/$(Hcasefilename)-timeseries.png"; xaxis=Hnames)
			end
		end
		dumpcsv = true
		if lon != nothing && lat != nothing
			if length(lon) != length(ch)
				plotmap = 1
			else
				NMFk.plot_wells("clusters-$(k).html", lon, lat, ch; figuredir=figuredir, hover=hover, title="Clusters: $k")
				lonlat = [lon lat]
				DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(map(i->"S$i", 1:k)) "Cluster"]; Hnames lonlat Hm ch], ',')
				dumpcsv = false
			end
		end
		if dumpcsv
			DelimitedFiles.writedlm("$resultdir/$(Hcasefilename)-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k)) "Cluster"]; Hnames Hm ch], ',')
		end
		if clusterattributes
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
 			elseif length(Wmap) > 0
 				@assert length(Wmap) == size(W[k], 1)
 				mu = unique(Ws)
 				na = length(mu)
 				@assert length(Wnames) == na
 				Wa = Matrix{eltype(W[k])}(undef, na, size(W[k], 2))
 				for (i, m) in enumerate(mu)
 					Wa[i,:] = sum(W[k][Wmap .== m,:]; dims=1)
 				end
			else
				Wa = W[k]
				@assert length(Wnames) == size(Wa, 1)
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
			cw = NMFk.labelassignements(NMFk.robustkmeans(permutedims(Wa), k; resultdir=resultdir, casefilename="Wmatrix", load=loadassignements, save=true)[1].assignments)
			@assert clusterlabels == sort(unique(cw))
			signalmap = NMFk.getsignalassignments(Wa[:,is], cw; clusterlabels=clusterlabels, dims=1)
			cassgined = zeros(Int64, length(Wnames))
			cnew = Vector{eltype(cw)}(undef, length(cw))
			cnew .= ' '
			snew = Vector{String}(undef, length(cw))
			Wm = Wa ./ maximum(Wa; dims=1)
			for (j, i) in enumerate(clusterlabels)
				iclustermap = signalmap[j]
				cnew[cw .== i] .= clusterlabels[iclustermap]
				snew[cw .== i] .= "S$(signalmap[j])"
				# @info "Signal $i -> S$(signalmap[j]) -> $(clusterlabels[signalmap[j]]) (k-means clustering)"
				ii = indexin(cw, [i]) .== true
				cassgined[ii] .+= 1
			end
			cs = sortperm(cnew)
			if any(cassgined .== 0)
				@warn "$(uppercasefirst(Wcasefilename)) not assigned to any cluster:"
				display(Wnames[cassgined .== 0])
			end
			if any(cassgined .> 1)
				@warn "$(uppercasefirst(Wcasefilename)) assigned to more than luster:"
				display([Wnames[cassgined .> 1] cassgined[cassgined .> 1]])
			end
			io = open("$resultdir/$(Wcasefilename)-groups-$(k).txt", "w")
			for (j, i) in enumerate(clusterlabels)
				@info "Signal $i (S$(signalmap[j]); k-means clustering; remapped)"
				write(io, "Signal $i (k-means clustering; remapped)\n")
				ii = indexin(cnew, [i]) .== true
				is = sortperm(Wm[ii,signalmap[j]]; rev=true)
				d = [Wnames[ii] Wm[ii,signalmap[j]]][is,:]
				display(d)
				for l in d
					write(io, l)
					write(io, '\n')
				end
				write(io, '\n')
			end
			close(io)
			# snew2 = copy(snew)
			# for i = 1:k
			# 	snew2[snew .== "S$(i)"] .= "S$(ws[i])"
			# end
			if createplots
				NMFk.plotmatrix(Wm; filename="$figuredir/$(Wcasefilename)-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				ws = sortperm(vec(sum(Wa; dims=1)); rev=true)
				NMFk.plotmatrix(Wm[:,ws]; filename="$figuredir/$(Wcasefilename)-signals-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				NMFk.plotmatrix(Wm[cs,is]; filename="$figuredir/$(Wcasefilename)-sorted-$(k).png", xticks=clustermap[is], yticks=["$(Wnames[cs][i]) $(cnew[cs][i])" for i=1:length(cw)], colorkey=false, quiet=false, minor_label_font_size=Wmatrix_font_size)
				# NMFk.plotmatrix(Wa./sum(Wa; dims=1); filename="$figuredir/$(Wcasefilename)-sum-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=1:length(cols)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				# NMFk.plotmatrix((Wa./sum(Wa; dims=1))[cs,:]; filename="$figuredir/$(Wcasefilename)-sum2-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[cs][i]) $(cw[cs][i])" for i=1:length(cols)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				NMFk.plotmatrix((Wa ./ sum(Wa; dims=1))[cs,is]; filename="$figuredir/$(Wcasefilename)-sorted-sumrows-$(k).png", xticks=clustermap[is], yticks=["$(Wnames[cs][i]) $(cnew[cs][i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
				NMFk.biplots(Wa ./ maximum(Wa), Wnames, collect(1:k); filename="$figuredir/$(Wcasefilename)-biplots-$(k)-original.pdf", background_color=background_color, types=cnew, plotlabel=Wplotlabel)
				NMFk.biplots(Wa[cs,is] ./ maximum(Wa), Wnames[cs], clustermap[is]; filename="$figuredir/$(Wcasefilename)-biplots-$(k).pdf", background_color=background_color, types=cnew[cs], plotlabel=Wplotlabel)
				length(Wtypes) > 0 && NMFk.biplots(Wa[cs,is] ./ maximum(Wa), Wnames[cs], clustermap[is]; filename="$figuredir/$(Wcasefilename)-biplots-type-$(k).pdf", background_color=background_color, colors=Wcolors[cs], plotlabel=Wplotlabel)
				if plottimeseries == :W
					Mads.plotseries(Wa ./ maximum(Wa), "$figuredir/$(Wcasefilename)-timeseries.png"; xaxis=Wnames)
				end
			end
			dumpcsv = true
			if lon != nothing && lat != nothing
				if length(lon) != length(cw)
					(plotmap == 1) && @warn("Coordinate data does not match the number of either W matrix rows or H matrix columns!")
				else
					NMFk.plot_wells("clusters-$(k).html", lon, lat, cw; figuredir=figuredir, hover=hover, title="Clusters: $k")
					lonlat = [lon lat]
					DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" "X" "Y" permutedims(map(i->"S$i", 1:k)) "Cluster"]; Wnames lonlat Wm cw], ',')
					dumpcsv = false
				end
			end
			if dumpcsv
				DelimitedFiles.writedlm("$resultdir/$(Wcasefilename)-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k)) "Cluster"];  Wnames Wm cw], ',')
			end
			if createplots
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
				if biplotcolor == :W
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))]
					biplotcolors = [typecolors(cnew, Wcolors); fill("gray", length(Hnames))]
				elseif biplotcolor == :WH
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))]
					biplotlabels = [typecolors(cnew, Wcolors); typecolors(ch, Hcolors[length(cnew)+1:end])]
				elseif biplotcolor == :H
					M = [permutedims(Ha ./ maximum(Ha)); Wa ./ maximum(Wa)]
					biplotcolors = [typecolors(ch, Hcolors); fill("gray", length(Wnames))]
				elseif biplotcolor == :none
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))]
					biplotcolors = [fill("blue", length(Wnames)); fill("red", length(Hnames))]
				end
				NMFk.biplots(M, biplotlabels, collect(1:k); filename="$figuredir/all-biplots-$(k)-original.pdf", background_color=background_color, colors=biplotcolors, plotlabel=biplotlabelflag, sortmag=false)
				if biplotcolor == :W
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,is]
					biplotcolors = [typecolors(cnew, Wcolors); fill("gray", length(Hnames))]
				elseif biplotcolor == :WH
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,is]
					biplotlabels = [typecolors(cnew, Wcolors); typecolors(ch, Hcolors[length(cnew)+1:end])]
				elseif biplotcolor == :H
					M = [permutedims(Ha ./ maximum(Ha)); Wa ./ maximum(Wa)][:,is]
					biplotcolors = [typecolors(ch, Hcasefilename); fill("gray", length(Wnames))]
				elseif biplotcolor == :none
					M = [Wa ./ maximum(Wa); permutedims(Ha ./ maximum(Ha))][:,is]
					biplotcolors = [fill("blue", length(Wnames)); fill("red", length(Hnames))]
				end
				NMFk.biplots(M, biplotlabels, clustermap[is]; filename="$figuredir/all-biplots-$(k).pdf", background_color=background_color, colors=biplotcolors, plotlabel=biplotlabelflag, sortmag=false)
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
			if lon != nothing && lat != nothing
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