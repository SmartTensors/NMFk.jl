import Gadfly
import Compose
import Images
import Colors
import DataFrames
import StatsBase
import Measures
import Mads

function typecolors(types::AbstractVector, colors::AbstractVector=NMFk.colors)
	ncolors = length(colors)
	ut = unique(types)
	c = length(ut)
	typecolors = Vector{String}(undef, length(types))
	if c <= ncolors
		for (j, t) in enumerate(ut)
			typecolors[types .== t] .= colors[j]
		end
	else
		@info("Number of colors ($(ncolors)) is less than the number of plotted attributes ($(c)); Gray color will be used!")
		typecolors .= "gray"
	end
	return typecolors
end

function biplots(X::AbstractMatrix, label::AbstractVector, mapping::AbstractVector=[]; hsize::Measures.AbsoluteLength=5Gadfly.inch, vsize::Measures.AbsoluteLength=5Gadfly.inch, quiet::Bool=false, figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", types=[], separate::Bool=false, typecolors=NMFk.colors, ncolors=length(colors), dpi=imagedpi, background_color=nothing, kw...)
	r, c = size(X)
	@assert length(label) == r
	@assert c > 1
	if length(types) > 0
		typecolors = NMFk.typecolors(types, colors)
	end
	if typecolors == NMFk.colors
		if c <= ncolors
			typecolors = NMFk.colors[1:c]
			ncolors = c
		else
			@info("Number of colors ($(ncolors)) is less than the number of plotted attributes ($(c))!")
		end
	else
		@assert length(typecolors) == r || length(typecolors) == c
	end
	if length(mapping) > 0
		@assert length(mapping) == c
		crange = sortperm(mapping)
	else
		crange = 1:c
	end
	rowp = Vector{Compose.Context}(undef, 0)
	for (j, c1) = enumerate(crange)
		colp = Vector{Union{Compose.Context,Gadfly.Plot}}(undef, 0)
		f = false
		for (i, c2) = enumerate(crange)
			i == j && continue
			if i < j
				push!(colp, Compose.compose(Compose.context(0, 0, 1Compose.w, 1Compose.h), Compose.fill(background_color), Compose.rectangle(0, 0, 1Compose.w, 1Compose.h)))
			else
				push!(colp, biplot(X ./ maximumnan(X), label, mapping; code=true, col1=c1, col2=c2, hsize=hsize, vsize=vsize, colors=typecolors, background_color=background_color, kw...))
				f = true
			end
			if separate && filename != ""
				pp = i < j ? biplot(X ./ maximumnan(X), label, mapping; code=true, col1=c1, col2=c2, hsize=hsize, vsize=vsize, colors=typecolors, background_color=background_color, kw...) : colp[end]
				ff = splitext(filename)
				filename_long = ff[1] * "_$(c1)_$(c2)" * ff[end]
				fl = joinpathcheck(figuredir, filename_long)
				plotfileformat(pp, fl, hsize, vsize; dpi=dpi)
			end
		end
		f && push!(rowp, Gadfly.hstack(colp...))
	end
	p = Gadfly.vstack(rowp...)
	!quiet && Mads.display(p; gwo=hsize * (c-1), gho=vsize * (c-1), gw=100Compose.mm * (c-1), gh=100Compose.mm * (c-1))
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		plotfileformat(p, j, hsize * (c-1), vsize * (c-1); dpi=dpi)
	end
	return nothing
end

function biplot(X::AbstractMatrix, label::AbstractVector, mapping::AbstractVector=[]; hsize::Measures.AbsoluteLength=5Gadfly.inch, vsize::Measures.AbsoluteLength=5Gadfly.inch, quiet::Bool=false, plotmethod::Symbol=:frame, plotline::Bool=false, plotlabel::Bool=!(length(label) > 100), figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", col1::Number=1, col2::Number=2, axisname::AbstractString="Signal", xtitle::AbstractString="$axisname $col1", ytitle::AbstractString="$axisname $col2", colors=NMFk.colors, ncolors=length(colors), gm=[], point_label_font_size=12Gadfly.pt, background_color=nothing, code::Bool=false, opacity::Number=1.0, dpi=imagedpi, sortmag::Bool=false, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)
	r, c = size(X)
	@assert length(label) == r
	@assert c > 1
	x = X[:,col1]
	y = X[:,col2]
	if length(mapping) > 0
		xtitle = "$axisname $(mapping[col1])"
		ytitle = "$axisname $(mapping[col2])"
	end
	if sortmag
		m = sum.(x.^2 .+ y.^2)
		iorder = sortperm(m; rev=true)
	else
		iorder = eachindex(x)
	end
	if plotmethod == :layers && r < 10000 # Gadfly fails if more than 10000 samples
		l = Vector{Vector{Gadfly.Layer}}(undef, 0)
		for i in iorder
			ic = (i - 1) % ncolors + 1
			plotline && push!(l, Gadfly.layer(x=[0, x[i]], y=[0, y[i]], Gadfly.Geom.line(), Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity))))
			if plotlabel
				push!(l, Gadfly.layer(x=[x[i]], y=[y[i]], label=[label[i]], Gadfly.Geom.point(), Gadfly.Geom.label(), Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity), highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, point_label_color=Colors.RGBA(parse(Colors.Colorant, colors[ic])))))
			else
				push!(l, Gadfly.layer(x=[x[i]], y=[y[i]], Gadfly.Geom.point(), Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity), highlight_width=0Gadfly.pt)))
			end
		end
		push!(l, Gadfly.layer(x=[1.], y=[1.], Gadfly.Geom.nil, Gadfly.Theme(point_size=0Gadfly.pt)))
		p = Gadfly.plot(l..., Gadfly.Theme(background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...)
	elseif plotmethod == :frame
		# palette = ncolors == length(x) ? Gadfly.parse_colorant(NMFk.colors) : Gadfly.parse_colorant(colors)
		cv = ncolors == length(x) ? vec(eachindex(x)) : colors
		l = Vector{Vector{Gadfly.Layer}}(undef, 0)
		if plotlabel
			inl = label .== ""
			if sum(inl) == length(x)
				plotlabel = false
				inext = Colon()
			else
				if sortmag
					m = sum.(x[inl].^2 .+ y[inl].^2)
					iorder1 = sortperm(m; rev=true)
				else
					iorder1 = 1:sum(inl)
				end
				inext = .!inl
			end
		else
			inext = Colon()
		end
		if sortmag
			m = sum.(x[inext].^2 .+ y[inext].^2)
			iorder2 = sortperm(m; rev=true)
		else
			st = ncolors == length(x) ? label[inext] : colors[inext]
			ust = unique(st)
			if length(ust) > 1 && length(ust) < length(x)
				cc = Vector{Int32}(undef, length(ust))
				for (i, s) in enumerate(ust)
					cc[i] = sum(st .== s)
				end
				io = sortperm(cc)
				nl = Vector{Int32}(undef, length(st))
				for (i, s) in enumerate(ust[io])
					nl[st .== s] .= i
				end
				iorder2 = sortperm(nl; rev=true)
			else
				iorder2 = Colon()
			end
		end
		dfw2 = DataFrames.DataFrame(x=x[inext][iorder2], y=y[inext][iorder2], label=label[inext][iorder2], color=cv[inext][iorder2])
		if plotlabel
			push!(l, Gadfly.layer(dfw2, x=:x, y=:y, label=:label, color=:color, Gadfly.Geom.point(), Gadfly.Geom.label(; position=:dynamic, hide_overlaps=true), Gadfly.Theme(point_size=point_size_label, highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size)))
		else
			push!(l, Gadfly.layer(dfw2, x=:x, y=:y, color=:color, Gadfly.Geom.point(), Gadfly.Theme(point_size=point_size_nolabel, highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size)))
		end
		if inext != Colon()
			dfw1 = DataFrames.DataFrame(x=x[inl][iorder1], y=y[inl][iorder1], label=label[inl][iorder1], color=cv[inl][iorder1])
			push!(l, Gadfly.layer(dfw1, x=:x, y=:y, color=:color, Gadfly.Geom.point(), Gadfly.Theme(point_size=point_size_nolabel, highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size)))
			palette = Gadfly.parse_colorant(vcat(colors[inext][iorder2], colors[inl][iorder1]))
		else
			palette = ncolors == length(x) ? Gadfly.parse_colorant(colors[inext][iorder2]) : Gadfly.parse_colorant(colors)
		end
		colormap = function(nc)
						palette[rem.((1:nc) .- 1, length(palette)) .+ 1]
					end
		p = Gadfly.plot(l..., Gadfly.Scale.color_discrete(colormap), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), gm...)
	else
		# palette = ncolors == length(x) ? Gadfly.parse_colorant(NMFk.colors) : Gadfly.parse_colorant(colors)
		palette = Gadfly.parse_colorant(colors)
		colormap = function(nc)
						palette[rem.((1:nc) .- 1, length(palette)) .+ 1]
					end
		cv = ncolors == length(x) ? vec(eachindex(x)) : colors
		if plotlabel
			p = Gadfly.plot([x y label cv], x=Gadfly.Col.value(1), y=Gadfly.Col.value(2), label=Gadfly.Col.value(3), color=Gadfly.Col.value(4), Gadfly.Geom.point(), Gadfly.Scale.color_discrete(colormap), Gadfly.Geom.label(; position=:dynamic, hide_overlaps=true), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), gm...)
		else
			p = Gadfly.plot([x y cv], x=Gadfly.Col.value(1), y=Gadfly.Col.value(2), color=Gadfly.Col.value(3), Gadfly.Geom.point(), Gadfly.Scale.color_discrete(colormap), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), gm...)
		end
	end
	if code
		return p
	end
	if !quiet
		Mads.display(p; gw=hsize, gh=vsize)
	end
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		plotfileformat(p, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end

function histogram(df::DataFrames.DataFrame, names=names(df); kw...)
	m = Matrix(df)
	m[ismissing.(m)] .= NaN
	histogram(convert.(Float64, m), names; kw...)
end

function histogram(data::AbstractMatrix, names::AbstractVector=["" for i in axes(data, 2)]; figuredir::AbstractString=".", filename_prefix::AbstractString="histogram", plot_type::AbstractString="png", save::Bool=false, save_data::Bool=false, quiet::Bool=false, kw...)
	@assert size(data, 2) == length(names)
	filename_plot = ""
	recursivemkdir(figuredir)
	vec_xmina = Vector{Vector{Float64}}(undef, size(data, 2))
	vec_xmaxa = Vector{Vector{Float64}}(undef, size(data, 2))
	vec_ya = Vector{Vector{Float64}}(undef, size(data, 2))
	for c in axes(data, 2)
		if names[c] == ""
			!quiet && @info("Histogram of Column $(c):")
			if figuredir != "." || save
				filename_plot = "$(filename_prefix)_column_$(c).$(plot_type)"
			end
			filename_data = save_data ? "$(filename_prefix)_column_$(c)_data" : ""
			vec_xmina[c], vec_xmaxa[c], vec_ya[c] = histogram(data[:,c]; figuredir=figuredir, kw..., filename_plot=filename_plot, filename_data=filename_data, quiet=quiet)
		else
			!quiet && @info("Histogram of attribute $(names[c]):")
			if figuredir != "." || save
				filename_plot = "$(filename_prefix)_$(names[c]).$(plot_type)"
			end
			filename_data = save_data ? "$(filename_prefix)_$(names[c])_data" : ""
			vec_xmina[c], vec_xmina[c], vec_ya[c] = histogram(data[:,c]; figuredir=figuredir, kw..., title=names[c], filename_plot=filename_plot, filename_data=filename_data, quiet=quiet)
		end
	end
	return vec_xmina, vec_xmaxa, vec_ya
end

function histogram(datain::AbstractVector; kw...)
	data = datain[.!isnan.(datain)]
	histogram(data, ones(Int8, length(data)); kw..., joined=false)
end

function histogram(data::AbstractVector, classes::AbstractVector; joined::Bool=true, separate::Bool=false, proportion::Bool=false, closed::Symbol=:left, hsize::Measures.AbsoluteLength=6Gadfly.inch, vsize::Measures.AbsoluteLength=4Gadfly.inch, quiet::Bool=false, debug::Bool=false, figuredir::AbstractString=".", filename_plot::AbstractString="", filename_data::AbstractString="", title::AbstractString="", xtitle::AbstractString="", ytitle::AbstractString="", ymin=nothing, ymax=nothing, xmin=nothing, xmax=nothing, gm=[], opacity::Number=joined ? 0.4 : 0.6, dpi=imagedpi, xmap=i->i, xlabelmap=nothing, edges=nothing, refine::Number=1, return_data::Bool=false)
	ndata = length(data)
	if ndata <= 1
		debug && @warn("Data input is too short to compute histogram (length of data = $ndata)!")
		return
	end
	@assert ndata == length(classes)
	mind = minimumnan(data)
	maxd = maximumnan(data)
	if isnothing(edges)
		histall = StatsBase.fit(StatsBase.Histogram, data; closed=closed)
	else
		histall = StatsBase.fit(StatsBase.Histogram, data, edges; closed=closed)
	end
	if all(histall.weights .== 0) # Fix for StatsBase bug
		s = typeof(histall.edges[1].step) <: AbstractFloat ? histall.edges[1].step : histall.edges[1].step.hi
		edges = histall.edges[1] .- s
		histall = StatsBase.fit(StatsBase.Histogram, data, edges; closed=closed)
	end
	# if typeof(histall.edges[1].step) <: Integer || typeof(histall.edges[1].step) <: AbstractFloat
	# 	newedges = histall.edges[1][1]:histall.edges[1].step/refine:histall.edges[1][end]
	# else
	# 	newedges = histall.edges[1][1]:histall.edges[1].step.hi/refine:histall.edges[1][end]
	# end
	newedges = collect(histall.edges...)
	xaxis = xmap.(newedges)
	dx = xaxis[2] - xaxis[1]
	if length(xaxis) > 2
		if closed == :left && maxd == xaxis[end-1]
			xmina = xaxis[begin:end-2]
			xmaxa = xaxis[2:end-1]
		elseif closed == :right && mind == xaxis[2]
			xmina = xaxis[2:end-1]
			xmaxa = xaxis[3:end]
		else
			xmina = xaxis[begin:end-1]
			xmaxa = xaxis[2:end]
		end
	else
		xmina = xaxis
		xmaxa = xaxis
	end
	xminl = isnothing(xmin) ? xmina[1] : min(xmina[1], xmin)
	xmaxl = isnothing(xmax) ? xmaxa[end] : max(xmaxa[end], xmax)
	l = []
	suc = sort(unique(classes))
	ccount = Vector{Int64}(undef, length(suc))
	vec_xmina = Vector{Vector{Float64}}(undef, length(suc))
	vec_xmaxa = Vector{Vector{Float64}}(undef, length(suc))
	vec_ya = Vector{Vector{Float64}}(undef, length(suc))
	local ymaxl = 0
	for (j, ct) in enumerate(suc)
		i = classes .== ct
		ccount[j] = sum(i)
		hist = StatsBase.fit(StatsBase.Histogram, data[i], newedges; closed=closed)
		y = proportion ? hist.weights ./ ndata : hist.weights
		if debug
			@info("Histogram weights:")
			display(y)
		end
		if length(xaxis) > 2
			if closed == :left && maxd == xaxis[end-1]
				ya = y[begin:end-1]
				ya[end] += y[end]
			elseif closed == :right && mind == xaxis[2]
				ya = y[2:end]
				ya[1] += y[1]
			else
				ya = y
			end
		else
			ya = y
		end
		ymaxl = max(maximum(ya), ymaxl)
		if length(ya) == 1
			ya = [0, ya[1], 0]
			xmina = [xmina[1] - dx, xmina...]
			xmaxa = [xmaxa..., xmaxa[end] + dx]
		end
		if debug
			@info("Histogram weights:")
			display([xmina xmaxa ya])
		end
		if filename_data != ""
			filename_data_long = joinpathcheck(figuredir, first(splitext(filename_data)) * "_$(ct).csv")
			if !quiet
				@info("Saving histogram data to file: $(filename_data_long)")
			end
			DelimitedFiles.writedlm(filename_data_long, [xmina xmaxa ya], ',')
		end
		ya = ya[1:length(xmina)] # Ensure that the length of ya matches xmina
		vec_xmina[j] = xmina
		vec_xmaxa[j] = xmaxa
		vec_ya[j] = ya
		push!(l, Gadfly.layer(xmin=xmina, xmax=xmaxa, y=ya, Gadfly.Geom.bar, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[j]), opacity))))
	end
	ymax = !isnothing(ymax) ? ymax : ymaxl
	s = [Gadfly.Coord.Cartesian(xmin=xminl, xmax=xmaxl, ymin=ymin, ymax=ymax), Gadfly.Scale.x_continuous(minvalue=xminl, maxvalue=xmaxl), Gadfly.Guide.xticks(ticks=collect(xminl:dx:xmaxl)), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...]
	if !isnothing(xlabelmap)
		s = [s..., Gadfly.Scale.x_continuous(minvalue=xminl, maxvalue=xmaxl, labels=xlabelmap)]
	end
	m = []
	if joined
		f = Gadfly.plot(l..., s..., Gadfly.Guide.title(title * ": Count $(ndata)"), Gadfly.Guide.manual_color_key("", ["Type $(suc[i]): $(ccount[i])" for i=eachindex(suc)], [colors[i] for i = eachindex(suc)]))
	else
		for (i, g) in enumerate(l)
			if title != ""
				mt = length(l) > 1 ? [Gadfly.Guide.title(title * " Type $(suc[i]) : $(ccount[i])")] : [Gadfly.Guide.title(title)]
			else
				mt = []
			end
			push!(m, Gadfly.plot(g, s..., mt...))
		end
		f = Gadfly.vstack(m...)
		vsize *= length(suc)
	end
	if filename_plot != ""
		filenamelong = joinpathcheck(figuredir, filename_plot)
		if separate && length(m) > 1
			vsize /= length(suc)
			fp = splitext(filenamelong)
			for (i, p) in enumerate(m)
				plotfileformat(p, join([fp[1], "_$i", fp[end]]), hsize, vsize; dpi=dpi)
				!quiet && Mads.display(p; gw=hsize, gh=vsize)
			end
		else
			plotfileformat(f, filenamelong, hsize, vsize; dpi=dpi)
			!quiet && Mads.display(f; gw=hsize, gh=vsize)
		end
	else
		!quiet && Mads.display(f; gw=hsize, gh=vsize)
	end
	if length(suc) > 1
		return vec_xmina, vec_xmaxa, vec_ya
	else
		return vec_xmina[1], vec_xmaxa[1], vec_ya[1]
	end
end

function plotscatter(df::DataFrames.DataFrame; quiet::Bool=false, hsize::Measures.AbsoluteLength=5Gadfly.inch, vsize::Measures.AbsoluteLength=5Gadfly.inch, figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", xtitle::AbstractString="", ytitle::AbstractString="", xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], dpi=imagedpi)
	nsignals = length(unique(sort(df[!, :Attribute])))
	loopcolors = nsignals + 1 > ncolors ? true : false
	if loopcolors
		tc = []
	else
		tc = [Gadfly.Scale.color_discrete_manual(colors[2:nsignals+1]...)]
	end
	# label="Well", Gadfly.Geom.point, Gadfly.Geom.label,
	ff = Gadfly.plot(Gadfly.layer(df, x="Truth", y="Prediction", color="Attribute", Gadfly.Theme(highlight_width=0Gadfly.pt)), Gadfly.layer(x=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], y=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], Gadfly.Geom.line(), Gadfly.Theme(line_width=4Gadfly.pt,default_color="red", discrete_highlight_color=c->nothing)), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc...)
	!quiet && Mads.display(ff; gw=hsize, gh=vsize)
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end

function plotscatter(x::AbstractArray, y::AbstractArray, aw...; kw...)
	plotscatter(vec(x), vec(y), aw...; kw...)
end

function plotscatter(x::AbstractVector, y::AbstractVector, color::AbstractVector=[], size::AbstractVector=[]; quiet::Bool=false, hsize::Measures.AbsoluteLength=5Gadfly.inch, vsize::Measures.AbsoluteLength=5Gadfly.inch, figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", xtitle::AbstractString="", ytitle::AbstractString="", line::Bool=false, xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, zmin=nothing, zmax=nothing, gm=[], point_size=2Gadfly.pt, key_position::Symbol=:right, keytitle="", colormap=Gadfly.Scale.lab_gradient("green","yellow","red"), polygon=nothing, point_color="red", line_color="gray", line_width::Measures.AbsoluteLength=2Gadfly.pt, dpi=imagedpi)
	if !isnothing(polygon)
		xmin = !isnothing(xmin) ? min(minimumnan(polygon[:,1]), xmin) : minimumnan(polygon[:,1])
		xmax = !isnothing(xmax) ? max(maximumnan(polygon[:,1]), xmax) : maximumnan(polygon[:,1])
		ymin = !isnothing(ymin) ? min(minimumnan(polygon[:,2]), ymin) : minimumnan(polygon[:,2])
		ymax = !isnothing(ymax) ? max(maximumnan(polygon[:,2]), ymax) : maximumnan(polygon[:,2])
		pm = [Gadfly.layer(x=polygon[:,1], y=polygon[:,2], Gadfly.Geom.polygon(preserve_order=true, fill=false), Gadfly.Theme(line_width=line_width, default_color=line_color))]
	else
		pm = []
	end
	if line
		m = [minimumnan([x y]), maximumnan([x y])]
		one2oneline = [Gadfly.layer(x=m, y=m, Gadfly.Geom.line(), Gadfly.Theme(line_width=line_width * 2, default_color=line_color))]
	else
		one2oneline = []
	end
	if length(size) == 0
		size = repeat([point_size], length(x))
	else
		@assert length(size) == length(x)
	end
	if length(color) > 0
		@assert length(color) == length(x)
		if eltype(color) <: Number
			if isnothing(zmax) && isnothing(zmin)
				vcolor = color
			else
				vcolor = copy(color)
			end
			if isnothing(zmin)
				zmin = minimumnan(color)
				vcolor = color
			else
				vcolor[vcolor .< zmin] .= zmin
			end
			if isnothing(zmax)
				zmax = maximumnan(color)
			else
				vcolor[vcolor .> zmax] .= zmax
			end
			zin = .!isnan.(vcolor)
			if sum(zin) == 0
				@warn("No valid values to plot! All values are NaN!")
				return nothing
			end
			ff = Gadfly.plot(Gadfly.layer(x=x[zin], y=y[zin], color=vcolor[zin], size=size[zin], Gadfly.Theme(highlight_width=0Gadfly.pt, default_color=point_color, point_size=point_size, key_position=key_position)), pm..., one2oneline..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Scale.color_continuous(minvalue=zmin, maxvalue=zmax, colormap=colormap), Gadfly.Guide.ColorKey(title=keytitle), Gadfly.Theme(key_position=key_position), gm...)
		else
			palette = Gadfly.parse_colorant(colors)
			colormap = function(nc)
							palette[rem.((1:nc) .- 1, length(palette)) .+ 1]
						end
			ff = Gadfly.plot(Gadfly.layer(x=x, y=y, color=color, size=size, Gadfly.Theme(highlight_width=0Gadfly.pt, default_color=point_color, point_size=point_size, key_position=key_position)), pm..., one2oneline..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Scale.color_discrete(colormap), Gadfly.Guide.ColorKey(title=keytitle), Gadfly.Theme(key_position=key_position), gm...)
		end
	else
		ff = Gadfly.plot(Gadfly.layer(x=x, y=y, size=size, Gadfly.Theme(highlight_width=0Gadfly.pt, default_color=point_color, point_size=point_size)), pm..., one2oneline..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Theme(key_position=key_position), gm...)
	end
	!quiet && Mads.display(ff; gw=hsize, gh=vsize)
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end

function plotbars(V::AbstractVector, A::AbstractVector; quiet::Bool=false, hsize::Measures.AbsoluteLength=8Gadfly.inch, vsize::Measures.AbsoluteLength=4Gadfly.inch, major_label_font_size=12Gadfly.pt, minor_label_font_size=10Gadfly.pt, figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", xtitle::AbstractString="", ytitle::AbstractString="", gm=[], dpi=imagedpi)
	nsignals = length(V)
	@assert nsignals == length(A)
	loopcolors = nsignals + 1 > ncolors ? true : false
	df = DataFrames.DataFrame()
	df[!, :Values] = V[end:-1:1]
	df[!, :Attributes] = A[end:-1:1]
	if loopcolors
		tc = []
	else
		tc = [Gadfly.Scale.color_discrete_manual(colors[nsignals+1:-1:2]...)]
	end
	ff = Gadfly.plot(df, x="Values", y="Attributes", color="Attributes", Gadfly.Geom.bar(position=:dodge, orientation=:horizontal), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), tc..., gm..., Gadfly.Theme(key_position=:none, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
	!quiet && Mads.display(ff; gw=hsize, gh=vsize)
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return ff
end

function plot2dmatrixcomponents(M::AbstractMatrix, dim::Integer=1; quiet::Bool=false, hsize::Measures.AbsoluteLength=8Gadfly.inch, vsize::Measures.AbsoluteLength=4Gadfly.inch, figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", xtitle::AbstractString="", ytitle::AbstractString="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false, otherdim::Integer=(dim == 1) ? 2 : 1, order::AbstractVector=sortperm(vec(maximum(M, otherdim))), dpi=imagedpi)
	msize = size(M)
	ndimensons = length(msize)
	@assert dim >= 1 && dim <= ndimensons
	nsignals = msize[dim]
	loopcolors = nsignals > ncolors ? true : false
	nx = dim == 1 ? msize[2] : msize[1]
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	componentnames = map(i->"T$i", 1:nsignals)
	pl = Vector{Any}(undef, nsignals)
	for i = 1:nsignals
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		if dim == 2
			pl[i] = Gadfly.layer(x=xvalues, y=M[:, order[i]], Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
		else
			pl[i] = Gadfly.layer(x=xvalues, y=M[order[i], :], Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
		end
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:nsignals])]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax), tc..., tx..., gm...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax), tc..., tx..., gm...)
	!quiet && Mads.display(ff; gw=hsize, gh=vsize)
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return ff
end

"Generate Sankey plots"
function sankey(c1::AbstractVector, c2::AbstractVector, t1::AbstractString, t2::AbstractString; filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end])
	s1 = length(unique(c1))
	s2 = length(unique(c2))
	n1 = ["$t1 $i" for i=1:s1]
	n2 = ["$t2 $i" for i=1:s2]
	nn = [n1; n2]
	ns = Vector{Int64}(undef, 0)
	nt = Vector{Int64}(undef, 0)
	v = Vector{Int64}(undef, 0)
	for i = 1:s1
		for j = 1:s2
			push!(ns, i - 1)
			push!(nt, s1 + j - 1)
			c = 0
			for k = eachindex(c1)
				if c1[k] == i && c2[k] == j
					c += 1
				end
			end
			push!(v, c)
		end
	end
	s = PlotlyJS.plot(PlotlyJS.sankey(node_label=nn, link_source=ns, link_target=nt, link_value=v))
	if filename != ""
		recursivemkdir(ffilenamen)
		PlotlyJS.savefig(p, filename; format=format)
	end
	return s
end
function sankey(c1::AbstractVector, c2::AbstractVector, c3::AbstractVector, t1::AbstractString, t2::AbstractString, t3::AbstractString; filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end])
	s1 = length(unique(c1))
	s2 = length(unique(c2))
	s3 = length(unique(c3))
	n1 = ["$t1 $i" for i=1:s1]
	n2 = ["$t2 $i" for i=1:s2]
	n3 = ["$t3 $i" for i=1:s3]
	nn = [n1; n2; n3]
	ns = Vector{Int64}(undef, 0)
	nt = Vector{Int64}(undef, 0)
	v = Vector{Int64}(undef, 0)
	for i = 1:s1
		for j = 1:s2
			push!(ns, i - 1)
			push!(nt, s1 + j - 1)
			c = 0
			for k = eachindex(c1)
				if c1[k] == i && c2[k] == j
					c += 1
				end
			end
			push!(v, c)
		end
	end
	for i = 1:s2
		for j = 1:s3
			push!(ns, s1 + i - 1)
			push!(nt, s1 + s2 + j - 1)
			c = 0
			for k = eachindex(c3)
				if c2[k] == i && c3[k] == j
					c += 1
				end
			end
			push!(v, c)
		end
	end
	s = PlotlyJS.plot(PlotlyJS.sankey(node_label=nn, link_source=ns, link_target=nt, link_value=v))
	if filename != ""
		recursivemkdir(ffilenamen)
		PlotlyJS.savefig(p, filename; format=format)
	end
	return s
end
function sankey(cc::AbstractVector, tt::AbstractVector; filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end])
	@assert length(cc) == length(tt)
	ss = Vector{Int64}(undef, length(cc))
	nn = Vector{Vector{String}}(undef, length(cc))
	for c = eachindex(cc)
		if c > 1
			@assert length(cc[c-1]) == length(cc[c])
		end
		ss[c] = length(unique(cc[c]))
		nn[c] = ["$(tt[c]) $i" for i=1:ss[c]]
	end
	nn = vcat(nn...)
	ns = Vector{Int64}(undef, 0)
	nt = Vector{Int64}(undef, 0)
	v = Vector{Int64}(undef, 0)
	local csum = 0
	for c = eachindex(cc)-1
		for i = 1:ss[c]
			for j = 1:ss[c+1]
				push!(ns, csum + i - 1)
				push!(nt, csum + ss[c] + j - 1)
				z = 0
				for k = eachindex(cc[c])
					if cc[c][k] == i && cc[c+1][k] == j
						z += 1
					end
				end
				push!(v, z)
			end
		end
		csum += ss[c]
	end
	s = PlotlyJS.plot(PlotlyJS.sankey(node_label=nn, link_source=ns, link_target=nt, link_value=v))
	if filename != ""
		recursivemkdir(ffilenamen)
		PlotlyJS.savefig(p, filename; format=format)
	end
	return s
end

function r2matrix(X::AbstractArray, Y::AbstractArray; normalize::Symbol=:none, kw...)
	D = Matrix{Float64}(undef, size(X, 2), size(Y, 2))
	for i in axes(Y, 2)
		for j in axes(X, 2)
			r2 = NMFk.r2(X[:,j], Y[:,i])
			D[j,i] = ismissing(r2) ? NaN : r2
		end
	end
	if normalize == :rows
		D ./= sum(D; dims=2)
	elseif normalize == :cols
		D ./= sum(D; dims=1)
	elseif normalize == :all
		NMFk.normalize!(D)
	end
	NMFk.plotmatrix(D; kw..., key_position=:none, quiet=false)
	return D
end

setplotfileformat = Mads.setplotfileformat
plotfileformat = Mads.plotfileformat