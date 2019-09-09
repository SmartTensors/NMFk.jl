import Gadfly
import Measures
import Colors
import Compose

function plotmatrix(X::AbstractVector; kw...)
	plotmatrix(convert(Array{Float64,2}, permutedims(X)); kw...)
end

function plotmatrix(X::AbstractMatrix; minvalue=minimumnan(X), maxvalue=maximumnan(X), key_tilte="", title="", xlabel="", ylabel="", xticks=nothing, yticks=nothing, xplot=nothing, yplot=nothing, xmatrix=nothing, ymatrix=nothing, gl=[], gm=[Gadfly.Guide.xticks(label=false, ticks=nothing), Gadfly.Guide.yticks(label=false, ticks=nothing)], masize::Int64=0, colormap=colormap_gyr, filename::String="", hsize=6Compose.inch, vsize=6Compose.inch, figuredir::String=".", colorkey::Bool=true, mask=nothing, dots=nothing, polygon=nothing, contour=nothing, linewidth::Measures.Length{:mm,Float64}=2Gadfly.pt, key_title_font_size=10Gadfly.pt, key_label_font_size=10Gadfly.pt, major_label_font_size=12Gadfly.pt, dotcolor="purple", linecolor="gray", defaultcolor=nothing, pointsize=1.5Gadfly.pt, dotsize=1.5Gadfly.pt, transform=nothing, code::Bool=false, nbins::Integer=0, flatten::Bool=false, rectbin::Bool=(nbins>0) ? false : true, dpi::Number=imagedpi)
	recursivemkdir(figuredir; filename=false)
	recursivemkdir(filename)
	Xp = deepcopy(min.(max.(movingwindow(X, masize), minvalue), maxvalue))
	if transform != nothing
		Xp = transform.(Xp)
	end
	nanmask!(Xp, mask)
	ys, xs, vs = Gadfly._findnz(x->!isnan(x), Xp)
	n, m = size(Xp)
	rect = checkrectbin(Xp)
	if xticks != nothing
		gm = [gm..., Gadfly.Scale.x_discrete(labels=i->xticks[i]), Gadfly.Guide.xticks(label=true)]
	end
	if yticks != nothing
		gm = [gm..., Gadfly.Scale.y_discrete(labels=i->yticks[i]), Gadfly.Guide.yticks(label=true)]
	end
	cs = colorkey ? [Gadfly.Guide.ColorKey(title=key_tilte)] : [Gadfly.Theme(key_position = :none)]
	cm = colormap == nothing ? [] : [Gadfly.Scale.ContinuousColorScale(colormap..., minvalue=minvalue, maxvalue=maxvalue)]
	cs = colormap == nothing ? [] : cs
	ds = min.(size(Xp)) == 1 ? [Gadfly.Scale.x_discrete, Gadfly.Scale.y_discrete] : []
	if polygon != nothing
		if xplot == nothing && yplot == nothing
			xplot = Vector{Float64}(undef, 2)
			xplot[1] = minimum(polygon[:,1])
			xplot[2] = maximum(polygon[:,1])
			yplot = Vector{Float64}(undef, 2)
			yplot[1] = minimum(polygon[:,2])
			yplot[2] = maximum(polygon[:,2])
		else
			xplot[1] = min(xplot[1], minimum(polygon[:,1]))
			xplot[2] = max(xplot[2], maximum(polygon[:,1]))
			yplot[1] = min(yplot[1], minimum(polygon[:,2]))
			yplot[2] = max(yplot[2], maximum(polygon[:,2]))
		end
	end
	if rectbin && !rect
		yflip = true
		xmin = 0.5
		xmax = m + 0.5
		ymin = 0.5
		ymax = n + 0.5
	else
		xmatrixmin = 0; xmatrixmax = m; ymatrixmin = 0; ymatrixmax = n;
		yflip = false
		sx = m
		sy = n
		if xmatrix != nothing
			xmatrixmin = xmatrix[1]; xmatrixmax = xmatrix[2];
			sx = xmatrixmax - xmatrixmin
		end
		if ymatrix != nothing
			ymatrixmin = ymatrix[1]; ymatrixmax = ymatrix[2]; yflip = false
			sy = ymatrixmax - ymatrixmin
		end
		dx = sx / m
		dy = sy / n
		xs = (xs .- 1) ./ m * sx .+ xmatrixmin
		ys = (-1 .- ys) ./ n * sy .+ ymatrixmax
		xmin = xmatrixmin + dx / 2
		xmax = xmatrixmax + dx / 2
		ymin = ymatrixmin + dy / 2
		ymax = ymatrixmax + dy / 2
		if polygon != nothing
			xmin = min(xplot[1], xmin)
			xmax = max(xplot[2], xmax)
			ymin = min(yplot[1], ymin)
			ymax = max(yplot[2], ymax)
		end
		if rect
			xrectmin = xs .- dx / 2
			xrectmax = xs .+ dx / 2
			yrectmin = ys .- dy / 2
			yrectmax = ys .+ dy / 2
		end
	end
	# @show ymatrixmin ymatrixmax xmatrixmax xmatrixmin
	gt = [Gadfly.Guide.title(title), Gadfly.Guide.xlabel(xlabel), Gadfly.Guide.ylabel(ylabel), Gadfly.Theme(major_label_font_size=major_label_font_size, key_label_font_size=key_label_font_size, key_title_font_size=key_title_font_size, bar_spacing=0Gadfly.mm), Gadfly.Coord.cartesian(yflip=yflip, fixed=true, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), Gadfly.Scale.x_continuous, Gadfly.Scale.y_continuous]
	if defaultcolor == nothing
		if length(vs) > 0
			if length(vs) < m * n && !rectbin
				l = [Gadfly.layer(x=xs, y=ys, color=vs, Gadfly.Theme(point_size=pointsize, highlight_width=0Gadfly.pt))]
			elseif rect
				l = [Gadfly.layer(x=xs, y=ys, color=vs, xmin=xrectmin, xmax=xrectmax, ymin=yrectmin, ymax=yrectmax, Gadfly.Geom.rect())]
			else
				l = [Gadfly.layer(x=xs, y=ys, color=vs, Gadfly.Geom.rectbin())]
			end
		else
			l = nothing
		end
	else
		if nbins == 0
			l = [Gadfly.layer(x=xs, y=ys, Gadfly.Theme(default_color=defaultcolor, point_size=pointsize, highlight_width=0Gadfly.pt))]
		else
			l = []
			s = (maxvalue - minvalue) / nbins
			s1 = minvalue
			s2 = minvalue + s
			for i = 1:nbins
				id = findall(i->(i > s1 && i <= s2), vs)
				c = Colors.RGBA(defaultcolor.r, defaultcolor.g, defaultcolor.b, defaultcolor.alpha/i)
				sum(id) > 0 && (l = [l..., Gadfly.layer(x=xs[id], y=ys[id], Gadfly.Theme(default_color=c, point_size=pointsize, highlight_width=0Gadfly.pt))])
				s1 += s
				s2 += s
			end
		end
	end
	if l == nothing && maxvalue != nothing && minvalue != nothing
		l = Gadfly.layer(x=[xmin, xmax], y=[ymin, ymax], color=[minvalue, maxvalue], Gadfly.Theme(point_size=0Gadfly.pt, highlight_width=0Gadfly.pt))
	end
	if polygon == nothing && contour == nothing && dots == nothing
		c = l..., gl..., ds..., cm..., cs..., gt..., gm...
	else
		c = []
		if polygon != nothing
			push!(c, Gadfly.layer(x=polygon[:,1], y=polygon[:,2], Gadfly.Geom.polygon(preserve_order=true, fill=false), Gadfly.Theme(line_width=linewidth, default_color=linecolor)))
		end
		if dots != nothing
			push!(c, Gadfly.layer(x=dots[:,1], y=dots[:,2], Gadfly.Theme(point_size=dotsize, highlight_width=0Gadfly.pt, default_color=dotcolor)))
		end
		if contour != nothing
			push!(c, Gadfly.layer(z=permutedims(contour .* (maxvalue - minvalue) .+ minvalue), x=collect(1:size(contour, 2)), y=collect(1:size(contour, 1)), Gadfly.Geom.contour(levels=[minvalue]), Gadfly.Theme(line_width=linewidth, default_color=linecolor)))
		end
		if l != nothing
			if mask != nothing
				c =  l..., gl..., ds..., cm..., cs..., gt..., gm..., c...
			else
				c = l..., gl..., ds..., cm..., cs..., gt..., gm..., c...
			end
		else
			c = c..., gl..., ds..., cm..., gt..., gm...
		end
	end
	p = Gadfly.plot(c...)
	# display(p); println();
	if filename != ""
		Gadfly.draw(Gadfly.PNG(joinpath(figuredir, filename), hsize, vsize, dpi=dpi), p)
		if flatten
			f = joinpath(figuredir, filename)
			e = splitext(f)
			cmd = `convert -background black -flatten -format jpg $f $(e[1]).jpg`
			run(pipeline(cmd, stdout=devnull, stderr=devnull))
			rm(f)
		end
	end
	if code
		return c
	else
		return p
	end
end

function checkrectbin(M::AbstractMatrix)
	xok = false
	iold = 0
	for i in sum(.!isnan.(M); dims=1)
		if iold != 0 && i != 0
			xok = true
			break
		else
			iold = i
		end
	end
	yok = false
	iold = 0
	for i in sum(.!isnan.(M); dims=2)
		if iold != 0 && i != 0
			yok = true
			break
		else
			iold = i
		end
	end
	return !(xok && yok)
end