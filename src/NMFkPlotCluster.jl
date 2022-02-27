import Distances
import Clustering
import StatsBase
import Gadfly
import Compose

struct HeatMapStatistic <: Gadfly.StatisticElement
	metric::Union{Nothing,Distances.Metric,Distances.SemiMetric}
	dim::Int64
end

heatmap(; metric::Union{Nothing,Distances.Metric,Distances.SemiMetric}=Distances.CosineDist(), dim::Int64=1) = HeatMapStatistic(metric, dim)

Gadfly.Stat.input_aesthetics(stat::HeatMapStatistic) = [:z, :x, :y]
Gadfly.Stat.output_aesthetics(stat::HeatMapStatistic) = [:xmin, :xmax, :ymin, :ymax]
Gadfly.Stat.default_scales(stat::HeatMapStatistic) = [Gadfly.Scale.z_func(), Gadfly.Scale.x_discrete(), Gadfly.Scale.y_discrete(), Gadfly.Scale.color_continuous()]

function Gadfly.Stat.apply_statistic(stat::HeatMapStatistic, scales::Dict{Symbol,Gadfly.ScaleElement}, coord::Gadfly.CoordinateElement, aes::Gadfly.Aesthetics)
	if stat.metric === nothing
		r, c = size(aes.z)
		aes.x = repeat(aes.x; inner=r)
		aes.y = repeat(aes.y; outer=c)
		dist = aes.z
		aes.color_key_title = ""
	else
		n = size(aes.z, stat.dim)
		aes.x = repeat(aes.x; outer=n)
		aes.y = repeat(aes.y; inner=n)
		dist = Distances.pairwise(stat.metric, aes.z; dims=stat.dim)
		if metric == Distances.CosineDist()
			dist[isnan.(dist)] .= 1
		else
			dist[isnan.(dist)] .= 0
		end
		aes.color_key_title = string(replace(split(string(typeof(stat.metric)), ".")[end], "Dist"=>""), "\n","distance")
	end
	Gadfly.Stat.apply_statistic(Gadfly.Stat.rectbin(), scales, coord, aes)
	color_scale = get(scales, :color, Gadfly.Scale.color_continuous)
	Gadfly.Scale.apply_scale(color_scale, [aes], Gadfly.Data(color=vec(dist)))
end

function branches(hc::Clustering.Hclust, location::Symbol, scaleheight::Number=0.1, height::Number=0)
	order = StatsBase.indexmap(hc.order)
	nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)
	branches_row = Vector{NTuple{2,Float64}}[]
	branches_col = Vector{NTuple{2,Float64}}[]
	ypos = 0.0
	useheight = height == 0
	userow = location == :both || location == :top
	usecol = location == :both || location == :right
	for i in 1:size(hc.merges, 1)
		x1, y1 = nodepos[hc.merges[i, 1]]
		x2, y2 = nodepos[hc.merges[i, 2]]

		xpos = (x1 + x2) / 2
		h = useheight ? hc.heights[i] * scaleheight : height
		ypos = max(y1, y2) + h

		nodepos[i] = (xpos, ypos)
		userow && push!(branches_row, [(x1,y1), (x1,ypos), (x2,ypos), (x2,y2)])
		usecol && push!(branches_col, [(y1,x1), (ypos,x1), (ypos,x2), (y2,x2)])
	end
	return branches_row, branches_col, ypos
end

struct Dendrogram <: Gadfly.GuideElement
	location::Symbol
	scaleheight::Number
	height::Number
	color::AbstractString
	linewidth::Measures.AbsoluteLength
	raw::Bool
	dim::Int64
	metric::Union{Distances.Metric,Distances.SemiMetric}
	linkage::Symbol
end

dendrogram(; location::Symbol=:both, scaleheight::Number=.1, height::Number=0.1, color::AbstractString="white", linewidth::Measures.AbsoluteLength=0.3Compose.pt, raw::Bool=true, dim::Int64=1, metric::Union{Distances.Metric,Distances.SemiMetric}=Distances.CosineDist(), linkage::Symbol=:complete) = Dendrogram(location, scaleheight, height, color, linewidth, raw, dim, metric, linkage)

function Gadfly.Guide.render(guide::Dendrogram, theme::Gadfly.Theme, aes::Gadfly.Aesthetics)
	userow = guide.location == :both || guide.location == :top
	usecol = guide.location == :both || guide.location == :right
	if guide.raw
		if userow
			dist = Distances.pairwise(guide.metric, aes.z; dims=2)
			if guide.metric == Distances.CosineDist()
				dist[isnan.(dist)] .= 1
			else
				dist[isnan.(dist)] .= 0
			end
			hc = Clustering.hclust(dist; linkage=guide.linkage)
			branches_row, _, pos_row = branches(hc, guide.location, guide.scaleheight, guide.height)
		end
		if usecol
			dist = Distances.pairwise(guide.metric, aes.z; dims=1)
			if guide.metric == Distances.CosineDist()
				dist[isnan.(dist)] .= 1
			else
				dist[isnan.(dist)] .= 0
			end
			hc = Clustering.hclust(dist; linkage=guide.linkage)
			_, branches_col, pos_col = branches(hc, guide.location, guide.scaleheight, guide.height)
		end
		r, c = size(aes.z)
	else
		dist = Distances.pairwise(guide.metric, aes.z; dims=guide.dim)
		if guide.metric == Distances.CosineDist()
			dist[isnan.(dist)] .= 1
		else
			dist[isnan.(dist)] .= 0
		end
		hc = Clustering.hclust(dist; linkage=guide.linkage)
		branches_row, branches_col, pos_col = branches(hc, guide.location, guide.scaleheight, guide.height)
		pos_row = pos_col
		r = c = size(aes.z, guide.dim)
	end
	gpg = Gadfly.Guide.PositionedGuide[]
	if userow
		ctx_row = Compose.context(units=Compose.UnitBox(0.5, pos_row, c, -pos_row, bottompad=4Gadfly.px), minheight=pos_row*25)
		Compose.compose!(ctx_row, Compose.line(branches_row), Compose.stroke(guide.color), Compose.linewidth(guide.linewidth))
		push!(gpg, Gadfly.Guide.PositionedGuide([ctx_row], 0, Gadfly.Guide.top_guide_position))
	end
	if usecol
		ctx_col = Compose.context(units=Compose.UnitBox(0, r+0.5, pos_col, -r, leftpad=4Gadfly.px), minwidth=pos_col*25)
		Compose.compose!(ctx_col, Compose.line(branches_col), Compose.stroke(guide.color), Compose.linewidth(guide.linewidth))
		push!(gpg, Gadfly.Guide.PositionedGuide([ctx_col], 0, Gadfly.Guide.right_guide_position))
	end
	return gpg
end

function plotdendrogram(X::AbstractMatrix; dim::Int64=1, metric=Distances.CosineDist(), metricheat=metric, linkage::Symbol=:complete, location::Symbol=:both, scaleheight::Number=.1, height::Number=0.1, color::AbstractString="white", linewidth::Measures.AbsoluteLength=0.3Compose.pt, xticks=nothing, yticks=nothing, minor_label_font_size=10Gadfly.pt, figuredir::AbstractString=".", filename::AbstractString="", dpi=imagedpi, hsize::Measures.AbsoluteLength=6Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch)
	gm = []
	r, c = size(X)
	if xticks === nothing
		push!(gm, Gadfly.Guide.xticks(label=false, ticks=nothing))
	else
		@assert length(xticks) == c
		push!(gm, Gadfly.Scale.x_discrete(labels=i->xticks[i]))
		push!(gm, Gadfly.Guide.xticks(label=true))
	end
	if yticks === nothing
		push!(gm, Gadfly.Guide.yticks(label=false, ticks=nothing))
	else
		@assert length(yticks) == r
		push!(gm, Gadfly.Scale.y_discrete(labels=i->yticks[i]))
		push!(gm, Gadfly.Guide.yticks(label=true))
	end
	raw = metricheat === nothing
	if raw
		r, c = size(X)
	else
		r = c = size(X, dim)
	end
	dist = Distances.pairwise(metric, X; dims=dim)
	dist[isnan.(dist)] .= 0
	if !LinearAlgebra.issymmetric(dist)
		@warn "Non symmetric matrix!"
		display(dist)
		return nothing
	end
	p = Gadfly.plot(z=X, x=1:c, y=1:r, heatmap(; metric=metricheat, dim=dim), Gadfly.Geom.rectbin(), Gadfly.Scale.color_continuous(colormap=Gadfly.Scale.lab_gradient("green","yellow","red")), Gadfly.Coord.cartesian(yflip=true, fixed=true), dendrogram(location=location, scaleheight=scaleheight, height=height, color=color, linewidth=linewidth, raw=raw, dim=dim, metric=metric, linkage=linkage), Gadfly.Guide.xlabel(""), Gadfly.Guide.ylabel(""), gm..., Gadfly.Theme(key_position=:none, minor_label_font_size=minor_label_font_size))
	Mads.display(p)
	if filename != ""
		j = joinpathcheck(figuredir, filename)
		recursivemkdir(j)
		plotfileformat(p, j, hsize, vsize; dpi=dpi)
	end
	return nothing
end