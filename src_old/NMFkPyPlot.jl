import PyPlot

function plotmatrix(A::AbstractMatrix, fig::PyPlot.Figure, x0::Number, y0::Number, pixelsize::Number; line_width::Number=2, alpha::Number=1)
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
	ax.plot([xl, xl], [yl, yr], "k", line_width=line_width)
	ax.plot([xl, .5 * pixelsize], [yl, yl], "k", line_width=line_width)
	ax.plot([xl, .5 * pixelsize], [yr, yr], "k", line_width=line_width)
	ax.plot([xr, xr], [yl, yr], "k", line_width=line_width)
	ax.plot([xr, w - .5 * pixelsize], [yl, yl], "k", line_width=line_width)
	ax.plot([xr, w - .5 * pixelsize], [yr, yr], "k", line_width=line_width)
	return ax, w, h
end

function plotequation(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, fig::PyPlot.Figure; x0::Number=-0.05, y0::Number=0.05, pixelsize::Number=0.10, alpha::Number=1)
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

function plotequation643(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, fig::PyPlot.Figure; x0::Number=-0.05, y0::Number=0.05, pixelsize::Number=0.12, alpha::Number=1)
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

function plotnmf(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix; filename::AbstractString="", movie::Bool=false, frame::Integer=0)
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
		@warn("Setting a new file name has failed!")
		return ""
	end
end
