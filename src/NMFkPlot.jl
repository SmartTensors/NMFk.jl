import PyPlot
import Gadfly
import Compose
import Images
import Colors
import DataFrames
import StatsBase

colors = ["red", "blue", "green", "orange", "magenta", "cyan", "brown", "pink", "lime", "navy", "maroon", "yellow", "olive", "springgreen", "teal", "coral", "#e6beff", "beige", "purple", "#4B6F44", "#9F4576"]
ncolors = length(colors)

function histogram(data::Vector, classes::Vector; joined::Bool=true, proportion::Bool=false, closed::Symbol=:left, quiet::Bool=false, hsize=8Gadfly.inch, vsize=6Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="Truth", ytitle::String="Prediction", ymin=nothing, ymax=nothing, gm=[], opacity::Number=0.3, dpi=imagedpi, xmap=i->i, xlabelmap=nothing, refine=1)
	ndata = length(data)
	@assert length(data) == length(classes)
	histall = StatsBase.fit(StatsBase.Histogram, data; closed=closed)
	newedges = histall.edges[1][1]:histall.edges[1].step.hi/refine:histall.edges[1][end]
	xaxis = xmap.(collect(newedges))
	xmin = minimum(xaxis)
	xmax = maximum(xaxis)
	l = []
	suc = sort(unique(classes))
	if !joined
		opacity = 0.9
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
	s = [Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Scale.x_continuous(minvalue=xmin, maxvalue=xmax), Gadfly.Guide.xticks(ticks=collect(xaxis)), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...]
	if xlabelmap != nothing
		s = [s..., Gadfly.Scale.x_continuous(minvalue=xmin, maxvalue=xmax, labels=xlabelmap)]
	end
	if joined
		f = Gadfly.plot(l..., s..., Gadfly.Guide.title(title * ": Count $(ndata)"), Gadfly.Guide.manual_color_key("", ["Type $(suc[i]): $(ccount[i])" for i=1:length(suc)], [colors[i] for i in suc]))
		!quiet && (display(f); println())
	else
		m = []
		for (i, g) in enumerate(l)
			push!(m, Gadfly.plot(g, s..., Gadfly.Guide.title(title * " Type $(suc[i]) : $(ccount[i])")))
		end
		f = Gadfly.vstack(m...)
		vsize *= length(suc)
		gw = Compose.default_graphic_width
		gh = Compose.default_graphic_height
		Compose.set_default_graphic_size(gw, gh * length(suc))
		!quiet && (display(f); println())
		Compose.set_default_graphic_size(gw, gh)
	end
	if filename != ""
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize), f)
	end
	return nothing
end

function plotscatter(df::DataFrames.DataFrame; quiet::Bool=false, hsize=8Gadfly.inch, vsize=6Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="Truth", ytitle::String="Prediction", xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], dpi=imagedpi)
	nfeatures = length(unique(sort(df[!, :Attribute])))
	loopcolors = nfeatures + 1 > ncolors ? true : false
	if loopcolors
		tc = []
	else
		tc = [Gadfly.Scale.color_discrete_manual(colors[2:nfeatures+1]...)]
	end
	# label="Well", Gadfly.Geom.point, Gadfly.Geom.label,
	ff = Gadfly.plot(Gadfly.layer(df, x="Truth", y="Prediction", color="Attribute", Gadfly.Theme(highlight_width=0Gadfly.pt)), Gadfly.layer(x=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], y=[minimum(df[!, :Truth]), maximum(df[!, :Truth])], Gadfly.Geom.line(), Gadfly.Theme(line_width=4Gadfly.pt,default_color="red")), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm..., tc...)
	gw = Compose.default_graphic_width
	gh = Compose.default_graphic_height
	Compose.set_default_graphic_size(gw, gw)
	!quiet && (display(ff); println())
	Compose.set_default_graphic_size(gw, gh)
	if filename != ""
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize), ff)
	end
	return nothing
end

function plotscatter(x::AbstractVector, y::AbstractVector; quiet::Bool=false, hsize=8Gadfly.inch, vsize=6Gadfly.inch, figuredir::String=".", filename::String="", title::String="", xtitle::String="Truth", ytitle::String="Prediction", xmin=nothing, xmax=nothing, ymin=nothing, ymax=nothing, gm=[], dpi=imagedpi)
	m = [min(x..., y...), max(x..., y...)]
	ff = Gadfly.plot(Gadfly.layer(x=x, y=y, Gadfly.Theme(highlight_width=0Gadfly.pt,default_color="red")), Gadfly.layer(x=m, y=m, Gadfly.Geom.line(), Gadfly.Theme(line_width=4Gadfly.pt,default_color="gray")), Gadfly.Coord.Cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Guide.title(title), Gadfly.Guide.XLabel(xtitle), Gadfly.Guide.YLabel(ytitle), gm...)
	gw = Compose.default_graphic_width
	gh = Compose.default_graphic_height
	Compose.set_default_graphic_size(gw, gw)
	!quiet && (display(ff); println())
	Compose.set_default_graphic_size(gw, gh)
	if filename != ""
		if !isdir(figuredir)
			mkdir(figuredir)
		end
		recursivemkdir(filename)
		Gadfly.draw(Gadfly.PDF(joinpath(figuredir, filename), hsize, vsize), ff)
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
		Gadfly.draw(Gadfly.PDF(joinpath(figuredir, filename), hsize, vsize), ff)
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
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=dpi), ff)
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

"Generate Sankey inputs"
function sankey(c1::AbstractVector, c2::AbstractVector, t1::AbstractString="Type1", t2::AbstractString="Type2")
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
	return nn, ns, nt, v
end
function sankey(c1::AbstractVector, c2::AbstractVector, c3::AbstractVector, t1::AbstractString="Type1", t2::AbstractString="Type2", t3::AbstractString="Type3")
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
	return nn, ns, nt, v
end

"""
Create directories recursively (if does not already exist)

$(DocumentFunction.documentfunction(recursivemkdir;
argtext=Dict("dirname"=>"directory")))
"""
function recursivemkdir(s::String; filename=true)
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
			Mads.madswarn("File $(sc) exists!")
			return
		elseif !isdir(sc)
			mkdir(sc)
			@info("Make dir $(sc)")
		end
	end
end