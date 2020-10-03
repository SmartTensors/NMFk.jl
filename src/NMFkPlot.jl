import PyPlot
import Gadfly
import Compose
import Images
import Colors
import DataFrames
import StatsBase
import Measures

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "#e6beff", "beige", "purple", "#4B6F44", "#9F4576"]
ncolors = length(colors)

function typecolors(types::AbstractVector, colors::AbstractVector=NMFk.colors)
	ncolors=length(colors)
	ut = unique(types)
	c = length(ut)
	typecolors = Vector{String}(undef, length(types))
	if c <= ncolors
		for (j, t) in enumerate(ut)
			typecolors[types .== t] .= NMFk.colors[j]
		end
	else
		@warn "Number of colors ($(ncolors)) is less than the number of plotted attributes ($(c))!"
		typecolors .= "gray"
	end
	return typecolors
end

function biplots(X::AbstractMatrix, label::AbstractVector, mapping::AbstractVector=[]; hsize=5Gadfly.inch, vsize=5Gadfly.inch, quiet::Bool=false, figuredir::String=".", filename::String="", title::String="", types=[], typecolors=NMFk.colors, ncolors=length(colors), dpi=imagedpi, background_color=nothing, kw...)
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
			@warn "Number of colors ($(ncolors)) is less than the number of plotted attributes ($(c))!"
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
		end
		f && push!(rowp, Gadfly.hstack(colp...))
	end
	p = Gadfly.vstack(rowp...)
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw * (c-1), gw * (c-1))
		display(p); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(p, j, hsize * (c-1), vsize * (c-1); dpi=dpi)
	end
	return nothing
end

function biplot(X::AbstractMatrix, label::AbstractVector, mapping::AbstractVector=[]; hsize=5Gadfly.inch, vsize=5Gadfly.inch, quiet::Bool=false, plotmethod::Symbol=:layers, plotline::Bool=false, plotlabel::Bool=!(length(label) > 100), figuredir::String=".", filename::String="", title::String="", col1::Number=1, col2::Number=2, axisname::String="Signal", xtitle::String="$axisname $col1", ytitle::String="$axisname $col2", colors=NMFk.colors, ncolors=length(colors), gm=[], point_label_font_size=12Gadfly.pt, background_color=nothing, code::Bool=false, opacity::Number=1.0, dpi=imagedpi, sortmag::Bool=true)
	r, c = size(X)
	@assert length(label) == r
	@assert c > 1
	x = X[:,col1]
	y = X[:,col2]
	if length(mapping) > 0
		xtitle = "$axisname $(mapping[col1])"
		ytitle = "$axisname $(mapping[col2])"
	end
	if plotmethod == :layers && r < 10000 # Gadfly fails if more than 10000 samples
		if sortmag
			m = sum.(x.^2 .+ y.^2)
			irange = sortperm(m; rev=true)
		else
			irange = 1:length(x)
		end
		l = Vector{Vector{Gadfly.Layer}}(undef, 0)
		for i in irange
			ic = (i - 1) % ncolors + 1
			plotline && push!(l, Gadfly.layer(x=[0, x[i]], y=[0, y[i]], Gadfly.Geom.line, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity))))
			if plotlabel
				push!(l, Gadfly.layer(x=[x[i]], y=[y[i]], label=[label[i]], Gadfly.Geom.point, Gadfly.Geom.label, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity), highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, point_label_color=Colors.RGBA(parse(Colors.Colorant, colors[ic])))))
			else
				push!(l, Gadfly.layer(x=[x[i]], y=[y[i]], Gadfly.Geom.point, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity), highlight_width=0Gadfly.pt)))
			end
		end
		push!(l, Gadfly.layer(x=[1.], y=[1.], Gadfly.Geom.nil, Gadfly.Theme(point_size=0Gadfly.pt)))
		p = Gadfly.plot(l..., Gadfly.Theme(background_color=background_color), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...)
	elseif plotmethod == :frame # very slow
		palette = Gadfly.parse_colorant(colors)
		colormap = function(nc)
						palette[rem.((1:nc) .- 1, length(palette)) .+ 1]
					end
		dfw = DataFrames.DataFrame(x=x, y=y, label=label)
		if plotlabel
			p = Gadfly.plot(dfw, x=:x, y=:y, label=:label, color=:label, Gadfly.Scale.color_discrete(colormap), Gadfly.Geom.point(), Gadfly.Geom.label(), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), gm...)
		else
			p = Gadfly.plot(dfw, x=:x, y=:y, color=:label, Gadfly.Scale.color_discrete(colormap), Gadfly.Geom.point(), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), gm...)
		end
	else
		palette = Gadfly.parse_colorant(colors)
		colormap = function(nc)
						palette[rem.((1:nc) .- 1, length(palette)) .+ 1]
					end
		if plotlabel
			p = Gadfly.plot([x y label], x=Gadfly.Col.value(1), y=Gadfly.Col.value(2), label=Gadfly.Col.value(3), color=Gadfly.Col.value(3), Gadfly.Scale.color_discrete(colormap), Gadfly.Geom.point(), Gadfly.Geom.label(; position=:dynamic, hide_overlaps=true), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), gm...)
		else
			p = Gadfly.plot([x y label], x=Gadfly.Col.value(1), y=Gadfly.Col.value(2), color=Gadfly.Col.value(3), Gadfly.Scale.color_discrete(colormap), Gadfly.Geom.point(), Gadfly.Theme(highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, background_color=background_color, key_position=:none), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(xmin=0, xmax=1, ymin=0, ymax=1), gm...)
		end

	end
	# display(p); println()
	if code
		return p
	end
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw, gw)
		display(p); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(p, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end

function histogram(datain::AbstractVector; kw...)
	data = datain[.!isnan.(datain)]
	histogram(data, ones(Int8, length(data)); kw..., opacity=0.6, joined=false)
end

function histogram(data::AbstractVector, classes::Vector; mergeedge::Bool=true, joined::Bool=true, separate::Bool=false, proportion::Bool=false, closed::Symbol=:left, hsize=6Gadfly.inch, vsize=4Gadfly.inch, quiet::Bool=false, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, xmin=nothing, xmax=nothing, gm=[], opacity::Number=0.6, dpi=imagedpi, xmap=i->i, xlabelmap=nothing, edges=nothing, refine::Number=1)
	ndata = length(data)
	@assert ndata == length(classes)
	if edges == nothing
		histall = StatsBase.fit(StatsBase.Histogram, data; closed=closed)
	else
		histall = StatsBase.fit(StatsBase.Histogram, data, edges; closed=closed)
	end
	if typeof(histall.edges[1].step) <: Integer || typeof(histall.edges[1].step) <: AbstractFloat
		newedges = histall.edges[1][1]:histall.edges[1].step/refine:histall.edges[1][end]
	else
		newedges = histall.edges[1][1]:histall.edges[1].step.hi/refine:histall.edges[1][end]
	end
	xaxis = xmap.(collect(newedges))
	dx = xaxis[2] - xaxis[1]
	if mergeedge
		if closed == :left
			xmina = xaxis[1:end-2]
			xmaxa = xaxis[2:end-1]
		else
			xmina = xaxis[2:end-1]
			xmaxa = xaxis[3:end]
		end
	else
		xmina = xaxis[1:end-1]
		xmaxa = xaxis[2:end]
	end
	xminl = xmin == nothing ? xmina[1] : min(xmina[1], xmin)
	xmaxl = xmax == nothing ? xmaxa[end] : max( xmaxa[end], xmax)
	l = []
	suc = sort(unique(classes))
	if !joined
		opacity = 0.6
	end
	local ymaxl = 0
	ccount = Vector{Int64}(undef, length(suc))
	for (j, ct) in enumerate(suc)
		i = classes .== ct
		ccount[j] = sum(i)
		hist = StatsBase.fit(StatsBase.Histogram, data[i], newedges; closed=closed)
		y = proportion ? hist.weights ./ ndata : hist.weights
		ymaxl = max(maximum(y), ymaxl)
		if mergeedge
			if closed == :left
				xmina = xaxis[1:end-2]
				xmaxa = xaxis[2:end-1]
				ya = y[1:end-1]
				ya[end] += y[end]
			else
				xmina = xaxis[2:end-1]
				xmaxa = xaxis[3:end]
				ya = y[2:end]
				ya[1] += y[1]
			end
		else
				xmina = xaxis[1:end-1]
				xmaxa = xaxis[2:end]
				ya = y
		end
		push!(l, Gadfly.layer(xmin=xmina, xmax=xmaxa, y=ya, Gadfly.Geom.bar, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[j]), opacity))))
	end
	ymax = ymax != nothing ? ymax : ymaxl
	s = [Gadfly.Coord.Cartesian(xmin=xminl, xmax=xmaxl, ymin=ymin, ymax=ymax), Gadfly.Scale.x_continuous(minvalue=xminl, maxvalue=xmaxl), Gadfly.Guide.xticks(ticks=collect(xminl:dx:xmaxl)), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...]
	if xlabelmap != nothing
		s = [s..., Gadfly.Scale.x_continuous(minvalue=xminl, maxvalue=xmaxl, labels=xlabelmap)]
	end
	m = []
	if joined
		f = Gadfly.plot(l..., s..., Gadfly.Guide.title(title * ": Count $(ndata)"), Gadfly.Guide.manual_color_key("", ["Type $(suc[i]): $(ccount[i])" for i=1:length(suc)], [colors[i] for i in 1:length(suc)]))
		!quiet && (display(f); println())
	else
		for (i, g) in enumerate(l)
			if title != ""
				if length(l) > 1
					mt = [Gadfly.Guide.title(title * " Type $(suc[i]) : $(ccount[i])")]
				else
					mt = [Gadfly.Guide.title(title)]
				end
			else
				mt = []
			end
			push!(m, Gadfly.plot(g, s..., mt...))
		end
		f = Gadfly.vstack(m...)
		vsize *= length(suc)
		if !quiet
			gw = Compose.default_graphic_width
			gh = Compose.default_graphic_height
			Compose.set_default_graphic_size(gw, gh * length(suc))
			display(f); println()
			Compose.set_default_graphic_size(gw, gh)
		end
	end
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		if separate && length(m) > 1
			vsize /= length(suc)
			fp = splitext(filename)
			for (i, p) in enumerate(m)
				plotfileformat(p, joinpath(figuredir, join([fp[1], "_$i", fp[end]])), hsize, vsize; dpi=dpi)
			end
		else
			plotfileformat(f, j, hsize, vsize; dpi=dpi)
		end
	end
	return nothing
end

function plotscatter(df::DataFrames.DataFrame; quiet::Bool=false, hsize=5Gadfly.inch, vsize=5Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], dpi=imagedpi)
	nfeatures = length(unique(sort(df[!, :Attribute])))
	loopcolors = nfeatures + 1 > ncolors ? true : false
	if loopcolors
		tc = []
	else
		tc = [Gadfly.Scale.color_discrete_manual(colors[2:nfeatures+1]...)]
	end
	# label="Well", Gadfly.Geom.point, Gadfly.Geom.label,
	ff = Gadfly.plot(Gadfly.layer(df, x="Truth", y="Prediction", color="Attribute", Gadfly.Theme(highlight_width=0Gadfly.pt)), Gadfly.layer(x=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], y=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], Gadfly.Geom.line(), Gadfly.Theme(line_width=4Gadfly.pt,default_color="red", discrete_highlight_color=c->nothing)), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc...)
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw, gw)
		display(ff); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end

function plotscatter(x::AbstractVector, y::AbstractVector, color=[], size=nothing; quiet::Bool=false, hsize=5Gadfly.inch, vsize=5Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", line::Bool=false, xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], point_size=2Gadfly.pt, keytitle="", polygon=nothing, pointcolor="red", linecolor="gray", linewidth::Measures.Length{:mm,Float64}=2Gadfly.pt, dpi=imagedpi)
	if size == nothing
		size = repeat([point_size], length(x))
	end
	if length(color) > 0
		palette = Gadfly.parse_colorant(colors)
		colormap = function(nc)
						palette[rem.((1:nc) .- 1, length(palette)) .+ 1]
					end
		cm = [Gadfly.Scale.color_continuous(minvalue=nothing, maxvalue=nothing, colormap=Gadfly.Scale.lab_gradient("green","yellow","red")), Gadfly.Guide.ColorKey(title=keytitle)]
	else
		cm = []
	end
	if polygon != nothing
		xmin = xmin != nothing ? min(minimumnan(polygon[:,1]), xmin) : minimumnan(polygon[:,1])
		xmax = xmax != nothing ? max(maximumnan(polygon[:,1]), xmax) : maximumnan(polygon[:,1])
		ymin = ymin != nothing ? min(minimumnan(polygon[:,2]), ymin) : minimumnan(polygon[:,2])
		ymax = ymax != nothing ? max(maximumnan(polygon[:,2]), ymax) : maximumnan(polygon[:,2])
		pm = [Gadfly.layer(x=polygon[:,1], y=polygon[:,2], Gadfly.Geom.polygon(preserve_order=true, fill=false), Gadfly.Theme(line_width=linewidth, default_color=linecolor))]
	else
		pm = []
	end
	if line
		m = [minimumnan([x y]), maximumnan([x y])]
		one2oneline = [Gadfly.layer(x=m, y=m, Gadfly.Geom.line(), Gadfly.Theme(line_width=linewidth * 2, default_color=linecolor))]
	else
		one2oneline = []
	end
	ff = Gadfly.plot(Gadfly.layer(x=x, y=y, color=color, size=size, Gadfly.Theme(highlight_width=0Gadfly.pt, default_color=pointcolor, point_size=point_size)), pm...,one2oneline..., Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., cm...)
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw, gw)
		display(ff); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end

function plotbars(V::AbstractVector, A::AbstractVector; quiet::Bool=false, hsize=8Gadfly.inch, vsize=4Gadfly.inch, major_label_font_size=12Gadfly.pt, minor_label_font_size=10Gadfly.pt, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", gm=[], dpi=imagedpi)
	nfeatures = length(V)
	@assert nfeatures == length(A)
	loopcolors = nfeatures + 1 > ncolors ? true : false
	df = DataFrames.DataFrame()
	df[!, :Values] = V[end:-1:1]
	df[!, :Attributes] = A[end:-1:1]
	if loopcolors
		tc = []
	else
		tc = [Gadfly.Scale.color_discrete_manual(colors[nfeatures+1:-1:2]...)]
	end
	ff = Gadfly.plot(df, x="Values", y="Attributes", color="Attributes", Gadfly.Geom.bar(position=:dodge, orientation=:horizontal), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), tc..., gm..., Gadfly.Theme(key_position=:none, major_label_font_size=major_label_font_size, minor_label_font_size=minor_label_font_size))
	!quiet && (display(ff); println())
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return ff
end

function plot2dmatrixcomponents(M::Matrix, dim::Integer=1; quiet::Bool=false, hsize=8Gadfly.inch, vsize=4Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, gm=[], timescale::Bool=true, code::Bool=false, otherdim=(dim == 1) ? 2 : 1, order=sortperm(vec(maximum(M, otherdim))), dpi=imagedpi)
	msize = size(M)
	ndimensons = length(msize)
	@assert dim >= 1 && dim <= ndimensons
	nfeatures = msize[dim]
	loopcolors = nfeatures > ncolors ? true : false
	nx = dim == 1 ? msize[2] : msize[1]
	xvalues = timescale ? vec(collect(1/nx:1/nx:1)) : vec(collect(1:nx))
	componentnames = map(i->"T$i", 1:nfeatures)
	pl = Vector{Any}(undef, nfeatures)
	for i = 1:nfeatures
		cc = loopcolors ? parse(Colors.Colorant, colors[(i-1)%ncolors+1]) : parse(Colors.Colorant, colors[i])
		if dim == 2
			pl[i] = Gadfly.layer(x=xvalues, y=M[:, order[i]], Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
		else
			pl[i] = Gadfly.layer(x=xvalues, y=M[order[i], :], Gadfly.Geom.line(), Gadfly.Theme(line_width=2Gadfly.pt, default_color=cc))
		end
	end
	tx = timescale ? [] : [Gadfly.Coord.Cartesian(xmin=minimum(xvalues), xmax=maximum(xvalues))]
	tc = loopcolors ? [] : [Gadfly.Guide.manual_color_key("", componentnames, colors[1:nfeatures])]
	if code
		return [pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax), tc..., tx..., gm...]
	end
	ff = Gadfly.plot(pl..., Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), Gadfly.Coord.Cartesian(ymin=ymin, ymax=ymax), tc..., tx..., gm...)
	!quiet && (display(ff); println())
	if filename != ""
		j = joinpath(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(ff, j, hsize, vsize; dpi=dpi)
	end
	return ff
end

function plotmatrix(A::Matrix, fig::PyPlot.Figure, x0::Number, y0::Number, pixelsize::Number; linewidth::Number=2, alpha::Number=1)
	w = pixelsize * size(A, 2)
	h = pixelsize * size(A, 1)
	ax = fig.add_axes([x0, y0, w, h], frameon=false)
	ax.axis("off")
	PyPlot.register_cmap("RYG", PyPlot.ColorMap("RYG", [parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red")]))
	ax.imshow(A, interpolation="nearest", extent=[0, w, 0, h], cmap=PyPlot.ColorMap("RYG"), alpha=alpha)
	gap = pixelsize / 5

	xl = 0 - gap
	xr = w + gap
	yl = 0 - gap
	yr = h + gap
	ax.plot([xl, xl], [yl, yr], "k", linewidth=linewidth)
	ax.plot([xl, .5 * pixelsize], [yl, yl], "k", linewidth=linewidth)
	ax.plot([xl, .5 * pixelsize], [yr, yr], "k", linewidth=linewidth)
	ax.plot([xr, xr], [yl, yr], "k", linewidth=linewidth)
	ax.plot([xr, w - .5 * pixelsize], [yl, yl], "k", linewidth=linewidth)
	ax.plot([xr, w - .5 * pixelsize], [yr, yr], "k", linewidth=linewidth)
	return ax, w, h
end

function plotequation(X::Matrix, W::Matrix, H::Matrix, fig::PyPlot.Figure; x0::Number=-0.05, y0::Number=0.05, pixelsize::Number=0.10, alpha::Number=1)
	owh, oww = size(W)
	ohh, ohw = size(H)
	#fig[:text](x0, y0, "×", fontsize=75, va="center")
	ax, w, h = plotmatrix(X, fig, x0, y0, pixelsize; alpha=alpha) # why does not start at the "x" symbol above
	# ax.text(w / 2, h + 1.5 * pixelsize, "X", fontsize=50, va="center", ha="center")
	#ax.text(0, 0, "+", fontsize=75, va="center") # why it is plotted here?!

	ax.text(w + pixelsize / 2, 0.5 * h, "=", fontsize=50, va="center")
	# ax.text(w + pixelsize / 2, h + 1.5 * pixelsize, "=", fontsize=50, va="center")

	_, ww, hw = plotmatrix(W, fig, x0 + w - oww * pixelsize / 4, y0, pixelsize, alpha=alpha)
	# ax.text(x0 + w + ww + pixelsize * 0.75, h + 1.5 * pixelsize, "W", fontsize=50, va="center", ha="center")

	ax.text(x0 + (size(X, 2) + size(W, 2)) * pixelsize + 1 * pixelsize, 0.5 * h, "×", fontsize=50, va="center")
	# ax.text(x0 + (size(X, 2) + size(W, 2)) * pixelsize + 2.5 * pixelsize, h + 1.5 * pixelsize, "×", fontsize=50, va="center")

	_, wh, hh = plotmatrix(H, fig, x0 + w + ww - ohw * pixelsize / 4, y0 + (size(W, 1) - size(H, 1)) / 2 * pixelsize, pixelsize; alpha=alpha)
	# ax.text(x0 + w + ww + wh + pixelsize * 1.5, h + 1.5 * pixelsize, "H", fontsize=50, va="center", ha="center")
end

function plotequation643(X::Matrix, W::Matrix, H::Matrix, fig::PyPlot.Figure; x0::Number=-0.05, y0::Number=0.05, pixelsize::Number=0.12, alpha::Number=1)
	owh, oww = size(W)
	ohh, ohw = size(H)
	#fig[:text](x0, y0, "×", fontsize=75, va="center")
	ax, w, h = plotmatrix(X, fig, x0, y0, pixelsize; alpha=alpha) # why does not start at the "x" symbol above
	ax.text(w / 2, h + 1.5 * pixelsize, "X", fontsize=50, va="center", ha="center")
	#ax.text(0, 0, "+", fontsize=75, va="center") # why it is plotted here?!

	ax.text(w + pixelsize / 2, 0.5 * h, "=", fontsize=50, va="center")
	ax.text(w + pixelsize / 2, h + 1.5 * pixelsize, "=", fontsize=50, va="center")

	_, ww, hw = plotmatrix(W, fig, x0 + w - oww * pixelsize / 3, y0, pixelsize, alpha=alpha)
	ax.text(x0 + w + ww + pixelsize * 0.75, h + 1.5 * pixelsize, "W", fontsize=50, va="center", ha="center")

	ax.text(x0 + (size(X, 2) + size(W, 2)) * pixelsize + 2.5 * pixelsize, 0.5 * h, "×", fontsize=50, va="center")
	ax.text(x0 + (size(X, 2) + size(W, 2)) * pixelsize + 2.5 * pixelsize, h + 1.5 * pixelsize, "×", fontsize=50, va="center")

	_, wh, hh = plotmatrix(H, fig, x0 + w + ww - ohw * pixelsize / 1.8, y0 + (size(W, 1) - size(H, 1)) / 2 * pixelsize, pixelsize; alpha=alpha)
	ax.text(x0 + w + ww + wh + pixelsize * 1.5, h + 1.5 * pixelsize, "H", fontsize=50, va="center", ha="center")
end

function plotnmf(X::Matrix, W::Matrix, H::Matrix; filename::AbstractString="", movie::Bool=false, frame::Integer=0)
	nr, nk = size(W)
	nk, nc = size(H)
	fig, throwawayax = PyPlot.subplots(figsize=(16,9))
	fig.delaxes(throwawayax)
	s = maximum(W, dims=1)
	W = W ./ s
	H = H .* permutedims(s)
	PyPlot.register_cmap("RYG", PyPlot.ColorMap("RYG", [parse(Colors.Colorant, "green"), parse(Colors.Colorant, "yellow"), parse(Colors.Colorant, "red")]))
	#spatialax = fig[:add_axes]([0, 0, 1, 1], frameon=false)
	#spatialax[:imshow](rand(100, 100), extent=[0, 100, 0, 100], cmap=PyPlot.ColorMap("RYG"), alpha=0.7, interpolation="nearest")

	#spatialax = fig[:add_axes]([0, 0, .5, .5], frameon=false)
	#spatialax[:imshow](rand(100, 100), extent=[0, 100, 0, 100], cmap=PyPlot.ColorMap("RYG"), alpha=0.7, interpolation="nearest")
	if nr == 6 && nc == 4 && nk == 3
		plotequation643(X, W, H, fig)
	elseif nr == 20 && nc == 5 && nk == 2
		plotequation(X, W, H, fig; pixelsize=1/nr, x0=0.1, y0=0)
	end

	if movie
		filename = setnewfilename(filename, frame)
		if frame > 0
			fig.text(0.9, 0.1, "$(sprintf("Iteration: %04d", frame))", fontsize=16, va="center", ha="center")
		end
	end
	if filename != ""
		fig.savefig(filename)
		Base.display(Images.load(filename)); println()
	else
		Base.display(fig); println()
	end
	PyPlot.close(fig)
end

function setnewfilename(filename::AbstractString, frame::Integer=0; keyword::AbstractString="frame")
	dir = dirname(filename)
	fn = splitdir(filename)[end]
	fs = split(fn, ".")
	if length(fs) == 1
		root = fs[1]
		ext = ""
	else
		root = join(fs[1:end-1], ".")
		ext = fs[end]
	end
	if ext == ""
		ext = "png"
		fn = fn * "." * ext
	end
	if !occursin(keyword, fn)
		fn = root * "-$(keyword)0000." * ext
	end
	rtest = occursin(Regex(string("-", keyword, "[0-9]*[.].*\$")), fn)
	if rtest
		rm = match(Regex(string("-", keyword, "([0-9]*)[.](.*)\$")), fn)
		if frame == 0
			v = parse(Int, rm.captures[1]) + 1
		else
			v = frame
		end
		l = length(rm.captures[1])
		f = "%0" * string(l) * "d"
		filename = "$(fn[1:rm.offset-1])-$(keyword)$(sprintf(f, v)).$(rm.captures[2])"
		return joinpath(dir, filename)
	else
		@warn("setnewfilename failed!")
		return ""
	end
end

"Convert `@Printf.sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@Printf.sprintf($(args...)))

"Generate Sankey plots"
function sankey(c1::AbstractVector, c2::AbstractVector, t1::AbstractString, t2::AbstractString; htmlfile::AbstractString="", pdffile::AbstractString="")
	s1 = length(unique(c1))
	s2 = length(unique(c2))
	n1 = ["$t1 $i" for i=1:s1]
	n2 = ["$t2 $i" for i=1:s2]
	nn = [n1; n2]
	ns = Array{Int64}(undef, 0)
	nt = Array{Int64}(undef, 0)
	v = Array{Int64}(undef, 0)
	for i = 1:s1
		for j = 1:s2
			push!(ns, i - 1)
			push!(nt, s1 + j - 1)
			c = 0
			for k = 1:length(c1)
				if c1[k] == i && c2[k] == j
					c += 1
				end
			end
			push!(v, c)
		end
	end
	s = PlotlyJS.plot(PlotlyJS.sankey(node_label=nn, link_source=ns, link_target=nt, link_value=v))
	htmlfile !="" && PlotlyJS.savehtml(s, htmlfile, :remote)
	pdffile != "" && PlotlyJS.savefig(s, pdffile)
	return s
end
function sankey(c1::AbstractVector, c2::AbstractVector, c3::AbstractVector, t1::AbstractString, t2::AbstractString, t3::AbstractString; htmlfile::AbstractString="", pdffile::AbstractString="")
	s1 = length(unique(c1))
	s2 = length(unique(c2))
	s3 = length(unique(c3))
	n1 = ["$t1 $i" for i=1:s1]
	n2 = ["$t2 $i" for i=1:s2]
	n3 = ["$t3 $i" for i=1:s3]
	nn = [n1; n2; n3]
	ns = Array{Int64}(undef, 0)
	nt = Array{Int64}(undef, 0)
	v = Array{Int64}(undef, 0)
	for i = 1:s1
		for j = 1:s2
			push!(ns, i - 1)
			push!(nt, s1 + j - 1)
			c = 0
			for k = 1:length(c1)
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
			for k = 1:length(c3)
				if c2[k] == i && c3[k] == j
					c += 1
				end
			end
			push!(v, c)
		end
	end
	s = PlotlyJS.plot(PlotlyJS.sankey(node_label=nn, link_source=ns, link_target=nt, link_value=v))
	htmlfile !="" && PlotlyJS.savehtml(s, htmlfile, :remote)
	pdffile != "" && PlotlyJS.savefig(s, pdffile)
	return s
end
function sankey(cc::AbstractVector, tt::AbstractVector; htmlfile::AbstractString="", pdffile::AbstractString="")
	@assert length(cc) == length(tt)
	ss = Array{Int64}(undef, length(cc))
	nn = Array{Array{String}}(undef, length(cc))
	for c = 1:length(cc)
		if c > 1
			@assert length(cc[c-1]) == length(cc[c])
		end
		ss[c] = length(unique(cc[c]))
		nn[c] = ["$(tt[c]) $i" for i=1:ss[c]]
	end
	nn = vcat(nn...)
	ns = Array{Int64}(undef, 0)
	nt = Array{Int64}(undef, 0)
	v = Array{Int64}(undef, 0)
	local csum = 0
	for c = 1:length(cc)-1
		for i = 1:ss[c]
			for j = 1:ss[c+1]
				push!(ns, csum + i - 1)
				push!(nt, csum + ss[c] + j - 1)
				z = 0
				for k = 1:length(cc[c])
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
	htmlfile !="" && PlotlyJS.savehtml(s, htmlfile, :remote)
	pdffile != "" && PlotlyJS.savefig(s, pdffile)
	return s
end

function r2matrix(X::AbstractArray, Y::AbstractArray; normalize::Symbol=:none, kw...)
	D = Array{Float64}(undef, size(X, 2), size(Y, 2))
	for i = 1:size(Y, 2)
		for j = 1:size(X, 2)
			r2 = NMFk.r2(X[:,j], Y[:,i])
			D[j,i] = ismissing(r2) ? NaN : r2
		end
	end
	if normalize == :rows
		D ./= sum(D; dims=2)
	elseif normalize == :cols
		D ./= sum(D; dims=1)
	elseif normalize == :all
		D = NMFk.normalize!(D)[1]
	end
	display(NMFk.plotmatrix(D; kw..., key_position=:none))
	return D
end

"""
Set image file `format` based on the `filename` extension, or sets the `filename` extension based on the requested `format`. The default `format` is `PNG`. `SVG`, `PDF`, `ESP`, and `PS` are also supported.

$(DocumentFunction.documentfunction(setplotfileformat;
                                    argtext=Dict("filename"=>"output file name")))

Returns:

- output file name
- output plot format (`png`, `pdf`, etc.)
"""
function setplotfileformat(filename::String, format::String="PNG")
	d = splitdir(filename)
	root, extension = splitext(d[end])
	if extension == ""
		extension = lowercase(format)
		filename = joinpath(d[1], root * "." * extension)
	else
		format = uppercase(extension[2:end])
	end
	if format == "EPS"
		format = "PS"
	end
	return filename, Symbol(format)
end

function plotfileformat(p, filename::String, hsize, vsize; dpi=imagedpi)
	filename, format = setplotfileformat(filename)
	if format == :SVG
		Gadfly.draw(Gadfly.eval(format)(filename, hsize, vsize), p)
	elseif isdefined(Main, :Cairo)
		if format == :PNG
			Gadfly.draw(Gadfly.PNG(filename, hsize, vsize; dpi=dpi), p)
		else
			Gadfly.draw(Gadfly.eval(format)(filename, hsize, vsize), p)
		end
	end
end