import DelimitedFiles
import PlotlyJS
import Mads

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
function clusterresults(krange::Union{AbstractRange{Int},AbstractVector{Int64},Integer}, W::AbstractVector, H::AbstractVector, Wnames::AbstractVector, Hnames::AbstractVector; clusterattributes::Bool=true, loadassignements::Bool=true, sizeW::Integer=0, lon=nothing, lat=nothing, hover=nothing, figuredir::AbstractString=".", resultdir::AbstractString=".", casefilenameW::AbstractString="attributes", casefilenameH::AbstractString="locations", Htypes::AbstractVector=[], Wtypes::AbstractVector=[], locationcolors=NMFk.colors, attributecolors=NMFk.colors, background_color="black", plotlabelH::Bool=true, plotlabelW::Bool=true, plottimeseries::Symbol=:none, biplotlabel::Symbol=:W, biplotcolor::Symbol=:W, cutoff::Number=0, cutoff_s::Number=0, Wmatrix_font_size=10Gadfly.pt, Hmatrix_font_size=10Gadfly.pt)
	if length(Htypes) > 0
		if locationcolors == NMFk.colors
			locationcolors = Vector{String}(undef, length(Htypes))
			for (j, t) in enumerate(unique(Htypes))
				locationcolors[Htypes .== t] .= NMFk.colors[j]
			end
		end
		Hnametype = Hnames .* " " .* String.(Htypes)
	else
		Hnametype = Hnames
	end
	if length(Wtypes) > 0
		if attributecolors == NMFk.colors
			attributecolors = Vector{String}(undef, length(Wtypes))
			for (j, t) in enumerate(unique(Wtypes))
				attributecolor[attributetype .== t] .= NMFk.colors[j]
			end
		end
		Wnametype = Wnames .* " " .* String.(Wtypes)
	end
	for k in krange
		@assert length(Hnames) == size(H[k], 2)
		sizeW == 1 && (@assert length(Wnames) == size(W[k], 1))
		@info("Number of signals: $k")

		@info("$(uppercasefirst(casefilenameH)) (signals=$k)")
		recursivemkdir(resultdir; filename=false)
		DelimitedFiles.writedlm("$resultdir/Hmatrix-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k))]; Hnames permutedims(H[k])], ',')
		if cutoff > 0
			ia = (H[k] ./ maximum(H[k]; dims=2)) .> cutoff
			for i in 1:k
				@info "Signal $i (max-normalized elements > $cutoff)"
				display(Hnames[ia[i,:]])
			end
		end
		ch = NMFk.labelassignements(NMFk.robustkmeans(H[k], k; resultdir=resultdir, casefilename="Hmatrix", load=loadassignements, save=true)[1].assignments)
		cs = sortperm(ch)
		clusterlabels = sort(unique(ch))
		signalmap = NMFk.getsignalassignments(H[k], ch; clusterlabels=clusterlabels, dims=2)
		clustermap = Vector{Char}(undef, k)
		clustermap .= ' '
		io = open("$resultdir/$(casefilenameH)-groups-$(k).txt", "w")
		for (j, i) in enumerate(clusterlabels)
			@info "Signal $i (S$(signalmap[j])) (k-means clustering)"
			write(io, "Signal $i (S$(signalmap[j])) (k-means clustering)\n")
			display(Hnames[indexin(ch, [i]) .== true])
			for l in Hnames[indexin(ch, [i]) .== true]
				write(io, l)
				write(io, '\n')
			end
			write(io, '\n')
			clustermap[signalmap[j]] = i
		end
		close(io)
		is = signalmap
		@assert is == sortperm(clustermap)
		NMFk.plotmatrix(permutedims(H[k] ./ maximum(H[k]; dims=2)); filename="$figuredir/$(casefilenameH)-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Hnames[i]) $(ch[i])" for i=1:length(ch)], colorkey=false, minor_label_font_size=Hmatrix_font_size)
		NMFk.plotmatrix((permutedims(H[k] ./ maximum(H[k]; dims=2)))[cs,is]; filename="$figuredir/$(casefilenameH)-sorted-$(k).png", xticks=clustermap[is], yticks=["$(Hnames[cs][i]) $(ch[cs][i])" for i=1:length(ch)], colorkey=false, quiet=false, minor_label_font_size=Hmatrix_font_size)
		NMFk.plotmatrix(permutedims((H[k] ./ sum(H[k]; dims=2)))[cs,is]; filename="$figuredir/$(casefilenameH)-sorted-sumrows-$(k).png", xticks=clustermap[is], yticks=["$(Hnametype[cs][i]) $(ch[cs][i])" for i=1:length(ch)], colorkey=false, minor_label_font_size=Hmatrix_font_size)
		NMFk.biplots(permutedims(H[k]) ./ maximum(H[k]), Hnames, collect(1:k); filename="$figuredir/$(casefilenameH)-biplots-$(k)-original.pdf", background_color=background_color, types=ch, plotlabel=plotlabelH)
		NMFk.biplots(permutedims(H[k])[cs,is] ./ maximum(H[k]), Hnames[cs], clustermap[is]; filename="$figuredir/$(casefilenameH)-biplots-$(k).pdf", background_color=background_color, types=ch[cs], plotlabel=plotlabelH)
		length(Htypes) > 0 && NMFk.biplots(permutedims(H[k])[cs,is]./ maximum(H[k]), Hnames[cs], clustermap[is]; filename="$figuredir/$(casefilenameH)-biplots-type-$(k).pdf", background_color=background_color, colors=locationcolors[cs], plotlabel=plotlabelH)
		if plottimeseries == :H
			Mads.plotseries(permutedims(H[k] ./ maximum(H[k]; dims=2)), "$figuredir/$(casefilenameH)-timeseries.png"; xaxis=Hnames)
		end
		if lon != nothing && lat != nothing && length(lon) == length(ch)
			@show ch
			NMFk.plot_wells("clusters-$(k).html", lon, lat, ch; figuredir=figuredir, hover=hover, title="Clusters: $k")
			lonlat= [lon lat]
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k).csv", [["Name" "X" "Y" permutedims(map(i->"S$i", 1:k)) "Cluster"]; Hnames lonlat permutedims(H[k] ./ maximum(H[k]; dims=2)) ch], ',')
		else
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k)) "Cluster"]; Hnames permutedims(H[k] ./ maximum(H[k]; dims=2)) ch], ',')
		end
		if clusterattributes
			if sizeW > 1
				na = convert(Int64, size(W[k], 1) / sizeW)
				Wa = Matrix{eltype(W[k])}(undef, na, size(W[k], 2))
				@assert length(Wnames) == size(Wa, 1)
				i1 = 1
				i2 = sizeW
				for i = 1:na
					Wa[i,:] = sum(W[k][i1:i2,:]; dims=1)
					i1 += sizeW
					i2 += sizeW
 				end
			else
				Wa = W[k]
			end
			@info("$(uppercasefirst(casefilenameW)) (signals=$k)")
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
			for (j, i) in enumerate(clusterlabels)
				iclustermap = signalmap[j]
				cnew[cw .== i] .= clusterlabels[iclustermap]
				snew[cw .== i] .= "S$(signalmap[j])"
				# @info "Signal $i -> S$(signalmap[j]) -> $(clusterlabels[signalmap[j]]) (k-means clustering)"
				imt = indexin(cw, [i]) .== true
				cassgined[imt] .+= 1
				display(Wnames[imt])
			end
			cs = sortperm(cnew)
			if any(cassgined .== 0)
				@warn "$(uppercasefirst(casefilenameW)) not assigned to any cluster:"
				display(Wnames[cassgined .== 0])
			end
			if any(cassgined .> 1)
				@warn "$(uppercasefirst(casefilenameW)) assigned to more than luster:"
				display([Wnames[cassgined .> 1] cassgined[cassgined .> 1]])
			end
			io = open("$resultdir/$(casefilenameW)-groups-$(k).txt", "w")
			for i in clusterlabels
				@info "Signal $i (k-means clustering; remapped)"
				write(io, "Signal $i (k-means clustering; remapped)\n")
				display(Wnames[indexin(cnew, [i]) .== true])
				for a in Wnames[indexin(cnew, [i]) .== true]
					write(io, a)
					write(io, '\n')
				end
				write(io, '\n')
			end
			close(io)
			NMFk.plotmatrix(Wa ./ maximum(Wa); filename="$figuredir/$(casefilenameW)-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
			ws = sortperm(vec(sum(Wa; dims=1)); rev=true)
			# snew2 = copy(snew)
			# for i = 1:k
			# 	snew2[snew .== "S$(i)"] .= "S$(ws[i])"
			# end
			NMFk.plotmatrix((Wa ./ maximum(Wa; dims=1))[:,ws]; filename="$figuredir/$(casefilenameW)-signals-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
			NMFk.plotmatrix((Wa ./ maximum(Wa; dims=1))[cs,is]; filename="$figuredir/$(casefilenameW)-sorted-$(k).png", xticks=clustermap[is], yticks=["$(Wnames[cs][i]) $(cnew[cs][i])" for i=1:length(cw)], colorkey=false, quiet=false, minor_label_font_size=Wmatrix_font_size)
			# NMFk.plotmatrix(Wa./sum(Wa; dims=1); filename="$figuredir/$(casefilenameW)-sum-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[i]) $(cw[i])" for i=1:length(cols)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
			# NMFk.plotmatrix((Wa./sum(Wa; dims=1))[cs,:]; filename="$figuredir/$(casefilenameW)-sum2-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(Wnames[cs][i]) $(cw[cs][i])" for i=1:length(cols)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
			NMFk.plotmatrix((Wa ./ sum(Wa; dims=1))[cs,is]; filename="$figuredir/$(casefilenameW)-sorted-sumrows-$(k).png", xticks=clustermap[is], yticks=["$(Wnames[cs][i]) $(cnew[cs][i])" for i=1:length(cw)], colorkey=false, minor_label_font_size=Wmatrix_font_size)
			NMFk.biplots(Wa ./ maximum(Wa), Wnames, collect(1:k); filename="$figuredir/$(casefilenameW)-biplots-$(k)-original.pdf", background_color=background_color, types=cnew, plotlabel=plotlabelW)
			NMFk.biplots(Wa[cs,is] ./ maximum(Wa), Wnames[cs], clustermap[is]; filename="$figuredir/$(casefilenameW)-biplots-$(k).pdf", background_color=background_color, types=cnew[cs], plotlabel=plotlabelW)
			length(Wtypes) > 0 && NMFk.biplots(Wa[cs,is] ./ maximum(Wa), Wnames[cs], clustermap[is]; filename="$figuredir/$(casefilenameW)-biplots-type-$(k).pdf", background_color=background_color, colors=attributecolors[cs], plotlabel=plotlabelW)
			if plottimeseries == :W
				Mads.plotseries(Wa ./ maximum(Wa), "$figuredir/$(casefilenameW)-timeseries.png"; xaxis=Wnames)
			end
			if lon != nothing && lat != nothing && length(lon) == length(cw)
				NMFk.plot_wells("clusters-$(k).html", lon, lat, cw; figuredir=figuredir, hover=hover, title="Clusters: $k")
				lonlat= [lon lat]
				DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k).csv", [["Name" "X" "Y" permutedims(map(i->"S$i", 1:k)) "Cluster"]; Wnames lonlat (Wa ./ maximum(Wa; dims=1)) cw], ',')
			else
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k).csv", [["Name" permutedims(map(i->"S$i", 1:k)) "Cluster"];  Wnames (Wa ./ maximum(Wa; dims=1)) cw], ',')
			end
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
				M = [Wa ./ maximum(Wa); permutedims(H[k] ./ maximum(H[k]))]
				biplotcolors = [typecolors(cnew, attributecolors); fill("gray", length(Hnames))]
			elseif biplotcolor == :WH
				M = [Wa ./ maximum(Wa); permutedims(H[k] ./ maximum(H[k]))]
				biplotlabels = [typecolors(cnew, attributecolors); typecolors(ch, locationcolors[length(cnew)+1:end])]
			elseif biplotcolor == :H
				M = [permutedims(H[k] ./ maximum(H[k])); Wa ./ maximum(Wa)]
				biplotcolors = [typecolors(ch, locationcolors); fill("gray", length(Wnames))]
			elseif biplotcolor == :none
				M = [Wa ./ maximum(Wa); permutedims(H[k] ./ maximum(H[k]))]
				biplotcolors = [fill("blue", length(Wnames)); fill("red", length(Hnames))]
			end
			NMFk.biplots(M, biplotlabels, collect(1:k); filename="$figuredir/all-biplots-$(k)-original.pdf", background_color=background_color, colors=biplotcolors, plotlabel=biplotlabelflag, sortmag=false)
			if biplotcolor == :W
				M = [Wa ./ maximum(Wa); permutedims(H[k] ./ maximum(H[k]))][:,is]
				biplotcolors = [typecolors(cnew, attributecolors); fill("gray", length(Hnames))]
			elseif biplotcolor == :WH
				M = [Wa ./ maximum(Wa); permutedims(H[k] ./ maximum(H[k]))][:,is]
				biplotlabels = [typecolors(cnew, attributecolors); typecolors(ch, locationcolors[length(cnew)+1:end])]
			elseif biplotcolor == :H
				M = [permutedims(H[k] ./ maximum(H[k])); Wa ./ maximum(Wa)][:,is]
				biplotcolors = [typecolors(ch, locationcolors); fill("gray", length(Wnames))]
			elseif biplotcolor == :none
				M = [Wa ./ maximum(Wa); permutedims(H[k] ./ maximum(H[k]))][:,is]
				biplotcolors = [fill("blue", length(Wnames)); fill("red", length(Hnames))]
			end
			NMFk.biplots(M, biplotlabels, clustermap[is]; filename="$figuredir/all-biplots-$(k).pdf", background_color=background_color, colors=biplotcolors, plotlabel=biplotlabelflag, sortmag=false)
		end

		if cutoff_s > 0
			attributesl = sizeW > 1 ? repeat(Wnames; inner=sizeW) : Wnames
			Xe = W[k] * H[k]
			local table = Hnames
			local table2 = Hnames
			local table3 = Hnames
			for i = 1:k
				Xek = (W[k][:,i:i] * H[k][i:i,:]) ./ Xe
				Xekm = Xek .> cutoff_s
				o = findmax(Xek; dims=1)
				table = hcat(table, map(i->attributesl[i], map(i->o[2][i][1], 1:length(Hnames))))
				table2 = hcat(table2, map(i->attributesl[Xekm[:,i]], 1:length(Hnames)))
				table3 = hcat(table3, map(i->sum(Xekm[:,i]), 1:length(Hnames)))
			end
			if lon != nothing && lat != nothing
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_max.csv", [lonlat table], ',')
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_$(cutoff_s).csv", [lonlat table2], ';')
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_count_$(cutoff_s).csv", [lonlat table3], ',')
			else
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_max.csv", table, ',')
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_$(cutoff_s).csv", table2, ';')
				DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_count_$(cutoff_s).csv", table3, ',')
			end
			local table = attributesl
			local table2 = attributesl
			local table3 = attributesl
			for i = 1:k
				Xek = (W[k][:,i:i] * H[k][i:i,:]) ./ Xe
				Xekm = Xek .> cutoff_s
				o = findmax(Xek; dims=2)
				table = hcat(table, map(i->Hnames[i], map(i->o[2][i][2], 1:length(attributesl))))
				table2 = hcat(table2, map(i->Hnames[Xekm[i,:]], 1:length(attributesl)))
				table3 = hcat(table3, map(i->sum(Xekm[i,:]), 1:length(attributesl)))
			end
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k)-table_max.csv", table, ',')
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k)-table_$(cutoff_s).csv", table2, ';')
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k)-table_count_$(cutoff_s).csv", table3, ',')
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
	for (j, i) in enumerate(clusterlabels)
		Hclustermap[Hsignalmap[j]] = i
	end
	Wclusterlabels = NMFk.labelassignements(NMFk.robustkmeans(permutedims(W), k; resultdir=resultdir, casefilename=Wclusterlabelcasefilename, load=loadassignements, save=true)[1].assignments)
	@assert clusterlabels == sort(unique(Wclusterlabels))
	Wsignalmap = NMFk.getsignalassignments(W[:,Hsignalmap], Wclusterlabels; clusterlabels=clusterlabels, dims=1)
	Wclusterlabelsnew = Vector{eltype(Wclusterlabels)}(undef, length(Wclusterlabels))
	Wclusterlabelsnew .= ' '
	Wsignalmapnew = Vector{String}(undef, length(Wclusterlabels))
	for (j, i) in enumerate(clusterlabels)
		iclustermap = Wsignalmap[j]
		Wclusterlabelsnew[Wclusterlabels .== i] .= clusterlabels[iclustermap]
		Wsignalmapnew[Wclusterlabels .== i] .= "S$(Wsignalmap[j])"
	end
	return Wclusterlabelsnew, Wsignalmapnew, Hclusterlabels, Hclustermap, Hsignalmap
end