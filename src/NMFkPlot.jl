import PyPlot
import Gadfly
import Plotly
import PlotlyJS
import Compose
import Images
import Colors
import DataFrames
import StatsBase

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "#e6beff", "beige", "purple", "#4B6F44", "#9F4576"]
ncolors = length(colors)

function plotbis(X::AbstractMatrix, label::AbstractVector, mapping::AbstractVector=Vector{Bool}(undef, 0); ratiofix=nothing, hsize=5Gadfly.inch, vsize=5Gadfly.inch, quiet::Bool=false, figuredir::String=".", filename::String="", title::String="", colors=NMFk.colors, dpi=imagedpi, kw...)
	r, c = size(X)
	@assert length(label) == r
	@assert c > 1
	if ratiofix == nothing
		ratiofix = (1. + 1. / (c - 1))
	end
	if length(mapping) > 0
		crange = sortperm(mapping)
	else
		crange = 1:c
	end
	rowp = Vector{Compose.Context}(undef, 0)
	for j = crange
		colp = Vector{Gadfly.Plot}(undef, 0)
		for i = crange
			i == j && continue
			push!(colp, plotbi(X, label, mapping; code=true, col1=j, hsize=hsize, vsize=vsize, colors=colors, col2=i, kw...))
		end
		push!(rowp, Gadfly.hstack(colp...))
		# if !quiet
		# 	gw = Compose.default_graphic_width
		# 	gh = Compose.default_graphic_height
		# 	Compose.set_default_graphic_size(gw * (c-1), gw)
		# 	display(rowp[end]); println()
		# 	Compose.set_default_graphic_size(gw, gh)
		# end
	end
	p = Gadfly.vstack(rowp...)
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw * (c-1), gw * (c-1) * ratiofix)
		display(p); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		plotfileformat(p, joinpath(figuredir, filename), hsize * (c-1), vsize * (c-1) * ratiofix; dpi=dpi)
	end
	return nothing
end

function plotbi(X::AbstractMatrix, label::AbstractVector, mapping::AbstractVector=Vector{Bool}(undef, 0); hsize=5Gadfly.inch, vsize=5Gadfly.inch, quiet::Bool=false, figuredir::String=".", filename::String="", title::String="", col1::Number=1, col2::Number=2, axisname::String="Signal", xtitle::String="$axisname $col1", ytitle::String="$axisname $col2", colors=NMFk.colors, ncolors=length(colors), gm=[], point_label_font_size=12Gadfly.pt, background_color=nothing, code::Bool=false, opacity::Number=1.0, dpi=imagedpi)
	r, c = size(X)
	@assert length(label) == r
	@assert c > 1
	xm = maximum(X)
	x = X[:,col1] ./ xm
	y = X[:,col2] ./ xm
	m = sum.(x.^2 .+ y.^2)
	l = Vector{Vector{Gadfly.Layer}}(undef, 0)
	if length(mapping) > 0
		xtitle = "$axisname $(mapping[col1])"
		ytitle = "$axisname $(mapping[col2])"
	end
	for i = sortperm(m; rev=true)
		ic = (i - 1) % ncolors + 1
		push!(l, Gadfly.layer(x=[0, x[i]], y=[0, y[i]], Gadfly.Geom.line, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity))))
		push!(l, Gadfly.layer(x=[x[i]], y=[y[i]], label=[label[i]], Gadfly.Geom.point, Gadfly.Geom.label, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ic]), opacity), highlight_width=0Gadfly.pt, point_label_font_size=point_label_font_size, point_label_color=Colors.RGBA(parse(Colors.Colorant, colors[ic])))))
	end
	push!(l, Gadfly.layer(x=[1.], y=[1.], Gadfly.Geom.nil, Gadfly.Theme(point_size=0Gadfly.pt)))
	p = Gadfly.plot(l..., Gadfly.Theme(background_color=background_color), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...)
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
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		plotfileformat(p, joinpath(figuredir, filename), hsize, vsize; dpi=dpi)
	end
	return nothing
end

function histogram(data::Vector; kw...)
	histogram(data, ones(Int8, length(data)); kw..., opacity=0.6, joined=false)
end

function histogram(data::Vector, classes::Vector; joined::Bool=true, separate::Bool=false, proportion::Bool=false, closed::Symbol=:left, hsize=6Gadfly.inch, vsize=4Gadfly.inch, quiet::Bool=false, figuredir::String=".", filename::String="", title::String="", xtitle::String="", ytitle::String="", ymin=nothing, ymax=nothing, xmin=nothing, xmax=nothing, gm=[], opacity::Number=0.6, dpi=imagedpi, xmap=i->i, xlabelmap=nothing, refine=1)
	ndata = length(data)
	@assert length(data) == length(classes)
	histall = StatsBase.fit(StatsBase.Histogram, data; closed=closed)
	newedges = histall.edges[1][1]:histall.edges[1].step.hi/refine:histall.edges[1][end]
	xaxis = xmap.(collect(newedges))
	xminl = xmin == nothing ? minimum(xaxis) : min(minimum(xaxis), xmin)
	xmaxl = xmax == nothing ? maximum(xaxis) : max(maximum(xaxis), xmax)
	l = []
	suc = sort(unique(classes))
	if !joined
		opacity = 0.6
	end
	local ymaxl = 0
	ccount = Vector{Int64}(undef, length(suc))
	for (j, ct) in enumerate(suc)
		i = findall((in)(ct), classes)
		ccount[j] = length(i)
		hist = StatsBase.fit(StatsBase.Histogram, data[i], newedges; closed=closed)
		y = proportion ? hist.weights ./ ndata : hist.weights
		ymaxl = max(maximum(y), ymaxl)
		push!(l, Gadfly.layer(xmin=xaxis[1:end-1], xmax=xaxis[2:end], y=y, Gadfly.Geom.bar, Gadfly.Theme(default_color=Colors.RGBA(parse(Colors.Colorant, colors[ct]), opacity))))
	end
	ymax = ymax != nothing ? yman : ymaxl
	s = [Gadfly.Coord.Cartesian(xmin=xminl, xmax=xmaxl, ymin=ymin, ymax=ymax), Gadfly.Scale.x_continuous(minvalue=xminl, maxvalue=xmaxl), Gadfly.Guide.xticks(ticks=unique([xminl; collect(xaxis); xmaxl])), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...]
	if xlabelmap != nothing
		s = [s..., Gadfly.Scale.x_continuous(minvalue=xminl, maxvalue=xmaxl, labels=xlabelmap)]
	end
	m = []
	if joined
		f = Gadfly.plot(l..., s..., Gadfly.Guide.title(title * ": Count $(ndata)"), Gadfly.Guide.manual_color_key("", ["Type $(suc[i]): $(ccount[i])" for i=1:length(suc)], [colors[i] for i in suc]))
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
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		if separate && length(m) > 1
			vsize /= length(suc)
			fp = splitext(filename)
			for (i, p) in enumerate(m)
				plotfileformat(p, joinpath(figuredir, join([fp[1], "_$i", fp[end]])), hsize, vsize; dpi=dpi)
			end
		else
			plotfileformat(f, joinpath(figuredir, filename), hsize, vsize; dpi=dpi)
		end
	end
	return nothing
end

function plotscatter(df::DataFrames.DataFrame; quiet::Bool=false, hsize=5Gadfly.inch, vsize=5Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="Truth", ytitle::String="Prediction", xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], dpi=imagedpi)
	nfeatures = length(unique(sort(df[!, :Attribute])))
	loopcolors = nfeatures + 1 > ncolors ? true : false
	if loopcolors
		tc = []
	else
		tc = [Gadfly.Scale.color_discrete_manual(colors[2:nfeatures+1]...)]
	end
	# label="Well", Gadfly.Geom.point, Gadfly.Geom.label,
	ff = Gadfly.plot(Gadfly.layer(df, x="Truth", y="Prediction", color="Attribute", Gadfly.Theme(highlight_width=0Gadfly.pt)), Gadfly.layer(x=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], y=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], Gadfly.Geom.line(), Gadfly.Theme(line_width=4Gadfly.pt,default_color="red")), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc...)
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw, gw)
		display(ff); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		plotfileformat(ff, joinpath(figuredir, filename), hsize, vsize; dpi=dpi)
	end
	return nothing
end

function plotscatter(x::AbstractVector, y::AbstractVector; quiet::Bool=false, hsize=5Gadfly.inch, vsize=5Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="Truth", ytitle::String="Prediction", xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], dpi=imagedpi)
	m = [minimumnan([x y]), maximumnan([x y])]
	ff = Gadfly.plot(Gadfly.layer(x=x, y=y, Gadfly.Theme(highlight_width=0Gadfly.pt,default_color="red")), Gadfly.layer(x=m, y=m, Gadfly.Geom.line(), Gadfly.Theme(line_width=4Gadfly.pt,default_color="gray")), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...)
	if !quiet
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw, gw)
		display(ff); println()
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		plotfileformat(ff, joinpath(figuredir, filename), hsize, vsize; dpi=dpi)
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
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		plotfileformat(ff, joinpath(figuredir, filename), hsize, vsize; dpi=dpi)
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
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		plotfileformat(ff, joinpath(figuredir, filename), hsize, vsize; dpi=dpi)
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

"Convert `@sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@sprintf($(args...)))

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
			D[j,i] = NMFk.r2(X[:,j], Y[:,i])
		end
	end
	if normalize == :rows
		D ./= sum(D; dims=2)
	else normalize == :cols
		D ./= sum(D; dims=1)
	end
	NMFk.plotmatrix(NMFk.normalizematrix_total!(D)[1]; kw..., key_position=:none)
end

function plot_wells(wx, wy, c; hover=nothing)
	if hover != nothing
		@assert length(hover) == length(wx)
	end
	@assert length(wx) == length(wy)
	@assert length(wx) == length(c)
	wells = []
	for (j, i) in enumerate(sort(unique(c)))
		ic = c .== i
		if hover != nothing
			well_p = PlotlyJS.scatter(;x=wx[ic], y=wy[ic], hovertext=hover[ic], mode="markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6))
		else
			well_p = PlotlyJS.scatter(;x=wx[ic], y=wy[ic], mode="markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6))
		end
		push!(wells, well_p)
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_wells(wx, wy, wz, c; hover=nothing)
	if hover != nothing
		@assert length(hover) == length(wx)
	end
	@assert length(wx) == length(wy)
	@assert length(wx) == length(wz)
	@assert length(wx) == length(c)
	wells = []
	for (j, i) in enumerate(sort(unique(c)))
		ic = c .== i
		if hover != nothing
			well_p = PlotlyJS.scatter3d(;x=wx[ic], y=wy[ic], z=wz[ic], hovertext=hover[ic], mode="markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6))
		else
			well_p = PlotlyJS.scatter3d(;x=wx[ic], y=wy[ic], z=wz[ic], mode="markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6))
		end
		push!(wells, well_p)
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_heel_toe_bad(heel_x, heel_y, toe_x, toe_y, c; hover=nothing)
	wells = []
	for (j, i) in enumerate(sort(unique(c)))
		ic = c .== i
		hx = heel_x[ic]
		hy = heel_y[ic]
		tx = toe_x[ic]
		ty = toe_y[ic]
		for k = 1:length(hx)
			well_trace = PlotlyJS.scatter(;x=[hx[k], tx[k]], y=[ty[k], ty[k]], mode="lines+markers", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6), line=Plotly.attr(width=2, color=NMFk.colors[j]), transform=Plotly.attr(type="groupby", groups=fill(i, length(hx)), styles=Plotly.attr(target="$i $(sum(ic))")), color=NMFk.colors[j])
			push!(wells, well_trace)
		end
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_heel_toe(heel_x, heel_y, toe_x, toe_y, c; hover=nothing)
	if hover != nothing
		@assert length(hover) == length(heel_x)
	end
	@assert length(heel_x) == length(heel_y)
	@assert length(heel_x) == length(toe_x)
	@assert length(heel_x) == length(toe_y)
	@assert length(heel_x) == length(c)
	traces = []
	for (j,i) in enumerate(sort(unique(c)))
		ic = c .== i
		hx = heel_x[ic]
		hy = heel_y[ic]
		tx = toe_x[ic]
		ty = toe_y[ic]
		x = vec(hcat([[hx[i] tx[i] NaN] for i = 1:length(hx)]...))
		y = vec(hcat([[hy[i] ty[i] NaN] for i = 1:length(hy)]...))
		if hover != nothing
			h = vec(hcat([[hover[i] hover[i] NaN] for i = 1:length(hover)]...))
			well_trace = PlotlyJS.scatter(;x=x, y=y, hovertext=h, mode="lines+markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6), line=Plotly.attr(width=2, color=NMFk.colors[j]))
		else
			well_trace = PlotlyJS.scatter(;x=x, y=y, mode="lines+markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6), line=Plotly.attr(width=2, color=NMFk.colors[j]))
		end
		push!(traces, well_trace)
	end
	return convert(Array{typeof(traces[1])}, traces)
end

function plot_heel_toe(heel_x, heel_y, heel_z, toe_x, toe_y, toe_z, c; hover=nothing)
	if hover != nothing
		@assert length(hover) == length(heel_x)
	end
	@assert length(heel_x) == length(heel_y)
	@assert length(heel_x) == length(toe_x)
	@assert length(heel_x) == length(toe_y)
	@assert length(heel_x) == length(c)
	traces = []
	for (j,i) in enumerate(sort(unique(c)))
		ic = c .== i
		hx = heel_x[ic]
		hy = heel_y[ic]
		hz = heel_z[ic]
		tx = toe_x[ic]
		ty = toe_y[ic]
		tz = toe_z[ic]
		x = vec(hcat([[hx[i] tx[i] NaN] for i = 1:length(hx)]...))
		y = vec(hcat([[hy[i] ty[i] NaN] for i = 1:length(hy)]...))
		z = vec(hcat([[hz[i] tz[i] NaN] for i = 1:length(hz)]...))
		if hover != nothing
			h = vec(hcat([[hover[i] hover[i] NaN] for i = 1:length(hover)]...))
			well_trace = PlotlyJS.scatter3d(;x=x, y=y, z=z, hovertext=h, mode="lines", name="$i $(sum(ic))", marker_color=NMFk.colors[j], line=Plotly.attr(width=6, color=NMFk.colors[j]))
		else
			well_trace = PlotlyJS.scatter3d(;x=x, y=y, z=z, mode="lines", name="$i $(sum(ic))", marker_color=NMFk.colors[j], line=Plotly.attr(width=6, color=NMFk.colors[j]))
		end
		push!(traces, well_trace)
	end
	return convert(Array{typeof(traces[1])}, traces)
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
	if format == :PNG
		Gadfly.draw(Gadfly.PNG(filename, hsize, vsize; dpi=dpi), p)
	else
		Gadfly.draw(Gadfly.eval(format)(filename, hsize, vsize), p)
	end
end