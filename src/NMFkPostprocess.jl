import DelimitedFiles
import PlotlyJS

"""
cutoff::Number = .9, cutoff_s::Number = 0.95
"""
function clusterresults(nkrange, W, H, robustness, locations, attributes; clusterattributes::Bool=true, sizeW::Integer=0, lat=nothing, lon=nothing, cutoff::Number=0, cutoff_s::Number=0, figuredir::AbstractString=".", resultdir::AbstractString=".", casefilenameW::AbstractString="attributes", casefilenameH::AbstractString="locations")
	for k = NMFk.getks(nkrange, robustness[nkrange])
		@info("Number of signals: $k")

		@info("Locations (signals=$k)")
		DelimitedFiles.writedlm("$resultdir/Hmatrix-$(k).csv", H[k], ',')
		if cutoff > 0
			ia = (H[k] ./ maximum(H[k]; dims=2)) .> cutoff
			for i in 1:k
				@info "Signal $i (max-normalized elements > $cutoff)"
				display(locations[ia[i,:]])
			end
		end
		c = NMFk.letterassignements(NMFk.robustkmeans(H[k], k; resultdir=resultdir, casefilename=casefilenameH, load=true, save=true).assignments)
		cs = sortperm(c)
		cletters = sort(unique(c))
		Hs = Matrix{Float64}(undef, k, k)
		for (j, i) in enumerate(cletters)
			Hs[j,:] .= vec(Statistics.mean(H[k][:, c .== i]; dims=2))
		end
		smap = NMFk.finduniquesignalsbest(Hs)
		cmap = Vector{Char}(undef, k)
		cmap .= ' '
		for (j, i) in enumerate(cletters)
			@info "Signal $i(S$(smap[j])) (k-means clustering)"
			display(locations[indexin(c, [i]) .== true])
			cmap[smap[j]] = i
		end
		is = sortperm(cmap)
		NMFk.plotmatrix(permutedims(H[k]) ./ maximum(H[k]); filename="$figuredir/$(casefilenameH)-$(k).png", vsize=12Compose.inch, xticks=["S$i" for i=1:k], yticks=["$(locations[i]) $(c[i])" for i=1:length(c)], colorkey=false)
		NMFk.plotmatrix((permutedims(H[k]) ./ maximum(H[k]))[cs,is]; filename="$figuredir/$(casefilenameH)-sorted-$(k).png", xticks=cmap[is], yticks=["$(locations[cs][i]) $(c[cs][i])" for i=1:length(c)], colorkey=false)
		NMFk.plotmatrix(permutedims((H[k] ./ sum(H[k]; dims=2)))[cs,is]; filename="$figuredir/$(casefilenameH)-sorted-sumrows-$(k).png", xticks=cmap[is], yticks=["$(locations[cs][i]) $(c[cs][i])" for i=1:length(c)], colorkey=false)
		NMFk.plotbis(permutedims(H[k])[cs,is], locations[cs], cmap; filename="$figuredir/$(casefilenameH)-biplots-$(k).pdf", background_color="black")
		if lat != nothing && lon != nothing
			p = PlotlyJS.plot(NMFk.plot_wells(lat, lon, c), Plotly.Layout(title="Clusters: $k"))
			PlotlyJS.savehtml(p, "$figuredir/clusters-$(k).html", :remote)
			latlon = [lat lon]
		else
			latlon = Vector{Char}(undef, size(H[k], 2))
			latlon .= ' '
		end
		DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k).csv", [locations latlon permutedims(H[k] ./ maximum(H[k])) c], ',')

		if clusterattributes
			Wa = W[k]
			if sizeW > 1
				na = convert(Int64, size(W[k], 1) / sizeW)
				Wa = Matrix{typeof(W[k][1,1])}(undef, na, size(W[k], 2))
				i1 = 1
				i2 = sizeW
				for i = 1:na
					Wa[i,:] = sum(W[k][i1:i2,:]; dims=1)
					i1 += sizeW
					i2 += sizeW
 				end
			end
			@info("Attributes (signals=$k)")
			DelimitedFiles.writedlm("$resultdir/Wmatrix-$(k).csv", Wa, ',')
			if cutoff > 0
				ia = (Wa ./ maximum(Wa; dims=1)) .> cutoff
				for i in 1:k
					@info "Signal $i (max-normalized elements > $cutoff)"
					display(attributes[ia[:,i]])
				end
			end
			c = NMFk.letterassignements(NMFk.robustkmeans(permutedims(Wa), k; resultdir=resultdir, casefilename=casefilenameW, load=true, save=true).assignments)
			@assert cletters == sort(unique(c))
			for (j, i) in enumerate(cletters)
				Hs[j,:] .= vec(Statistics.mean(Wa[c .== i,is]; dims=1))
			end
			smap = NMFk.finduniquesignalsbest(Hs)
			cassgined = zeros(Int64, length(attributes))
			cnew = Vector{typeof(c[1])}(undef, length(c))
			cnew .= ' '
			for (j, i) in enumerate(cletters)
				icmap = smap[j]
				cnew[c .== i] .= cletters[icmap]
				@info "Signal $i -> S$(smap[j]) -> $(cletters[smap[j]]) (k-means clustering)"
				imt = indexin(c, [i]) .== true
				cassgined[imt] .+= 1
				display(attributes[imt])
			end
			if any(cassgined .== 0)
				@warn "Attributes not assigned to any cluster:"
				display(attributes[cassgined .== 0])
			end
			if any(cassgined .> 1)
				@warn "Attributes assigned to more than luster:"
				display([attributes[cassgined .> 1] cassgined[cassgined .> 1]])
			end
			for i in cletters
				@info "Signal $i (k-means clustering; remapped)"
				display(attributes[indexin(cnew, [i]) .== true])
			end
			cs = sortperm(cnew)
			NMFk.plotmatrix(Wa ./ maximum(Wa); filename="$figuredir/$(casefilenameW)-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(attributes[i]) $(c[i])" for i=1:length(c)], colorkey=false)
			NMFk.plotmatrix((Wa ./ maximum(Wa))[cs,is]; filename="$figuredir/$(casefilenameW)-sorted-$(k).png", xticks=cmap[is], yticks=["$(attributes[cs][i]) $(cnew[cs][i])" for i=1:length(c)], colorkey=false)
			# NMFk.plotmatrix(Wa./sum(Wa; dims=1); filename="$figuredir/$(casefilenameW)-sum-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(attributes[i]) $(c[i])" for i=1:length(cols)], colorkey=false)
			# NMFk.plotmatrix((Wa./sum(Wa; dims=1))[cs,:]; filename="$figuredir/$(casefilenameW)-sum2-$(k).png", xticks=["S$i" for i=1:k], yticks=["$(attributes[cs][i]) $(c[cs][i])" for i=1:length(cols)], colorkey=false)
			NMFk.plotmatrix((Wa ./ sum(Wa; dims=1))[cs,is]; filename="$figuredir/$(casefilenameW)-sorted-sumrows-$(k).png", xticks=cmap[is], yticks=["$(attributes[cs][i]) $(cnew[cs][i])" for i=1:length(c)], colorkey=false)
			NMFk.plotbis(Wa[cs,is], attributes[cs], cmap; filename="$figuredir/$(casefilenameW)-biplots-$(k).pdf", background_color="black")
			DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k).csv", [attributes (Wa ./ maximum(Wa))	 c], ',')
		end

		if cutoff_s > 0
			attributesl = sizeW > 1 ? repeat(attributes; inner=sizeW) : attributes
			Xe = W[k] * H[k]
			local table = locations
			local table2 = locations
			local table3 = locations
			for i = 1:k
				Xek = (W[k][:,i:i] * H[k][i:i,:]) ./ Xe
				Xekm = Xek .> cutoff_s
				o = findmax(Xek; dims=1)
				table = hcat(table, map(i->attributesl[i], map(i->o[2][i][1], 1:length(locations))))
				table2 = hcat(table2, map(i->attributesl[Xekm[:,i]], 1:length(locations)))
				table3 = hcat(table3, map(i->sum(Xekm[:,i]), 1:length(locations)))
			end
			DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_max.csv", [latlon table], ',')
			DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_$(cutoff_s).csv", [latlon table2], ';')
			DelimitedFiles.writedlm("$resultdir/$(casefilenameW)-$(k)-table_count_$(cutoff_s).csv", [latlon table3], ',')
			local table = attributesl
			local table2 = attributesl
			local table3 = attributesl
			for i = 1:k
				Xek = (W[k][:,i:i] * H[k][i:i,:]) ./ Xe
				Xekm = Xek .> cutoff_s
				o = findmax(Xek; dims=2)
				table = hcat(table, map(i->locations[i], map(i->o[2][i][2], 1:length(attributesl))))
				table2 = hcat(table2, map(i->locations[Xekm[i,:]], 1:length(attributesl)))
				table3 = hcat(table3, map(i->sum(Xekm[i,:]), 1:length(attributesl)))
			end
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k)-table_max.csv", table, ',')
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k)-table_$(cutoff_s).csv", table2, ';')
			DelimitedFiles.writedlm("$resultdir/$(casefilenameH)-$(k)-table_count_$(cutoff_s).csv", table3, ',')
		end
	end
end