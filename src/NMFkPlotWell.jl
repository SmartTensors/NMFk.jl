import Plotly
import PlotlyJS

function plot_wells(filename::AbstractString, v...; figuredir::AbstractString=".", title::AbstractString="", plotly=nothing, k...)
	if plotly == nothing
		p = PlotlyJS.plot(NMFk.plot_wells(v...; k...), Plotly.Layout(title=title, hovermode="closest", yaxis_scaleanchor="x", yaxis_scaleratio=1))
	else
		p = PlotlyJS.plot(plotly, Plotly.Layout(title=title, hovermode="closest", yaxis_scaleanchor="x", yaxis_scaleratio=1))
		p = Plotly.addtraces(p, NMFk.plot_wells(v...; k...)...)
	end
	j = joinpath(figuredir, filename)
	recursivemkdir(j)
	PlotlyJS.savehtml(p, j, :remote)
end

function plot_wells(wx::AbstractVector, wy::AbstractVector, c::AbstractVector; hover=nothing, label=nothing, pointsize=6)
	if hover != nothing
		@assert length(hover) == length(wx)
	end
	if label != nothing
		@assert length(label) == length(wx)
	end
	@assert length(wx) == length(wy)
	@assert length(wx) == length(c)
	wells = []
	for (j, i) in enumerate(unique(sort(c)))
		ic = c .== i
		l = label == nothing ? Dict(:mode=>"markers") : Dict(:mode=>"markers+text", :text=>label, :textposition=>"left center")
		h = hover == nothing ? Dict() : Dict(:hovertext=>hover[ic], :hoverinfo=>"text")
		well_p = PlotlyJS.scatter(;x=wx[ic], y=wy[ic], l..., name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(; size=pointsize), h...)
		push!(wells, well_p)
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_wells(wx::AbstractVector, wy::AbstractVector, wz::AbstractVector, c::AbstractVector; hover=nothing, pointsize=6)
	if hover != nothing
		@assert length(hover) == length(wx)
	end
	@assert length(wx) == length(wy)
	@assert length(wx) == length(wz)
	@assert length(wx) == length(c)
	wells = []
	for (j, i) in enumerate(unique(sort(c)))
		ic = c .== i
		h = hover == nothing ? Dict() : Dict(:hovertext=>hover[ic], :hoverinfo=>"text")
		well_p = PlotlyJS.scatter3d(;x=wx[ic], y=wy[ic], z=wz[ic], mode="markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(; size=pointsize), h...)
		push!(wells, well_p)
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_heel_toe_bad(heel_x::AbstractVector, heel_y::AbstractVector, toe_x::AbstractVector, toe_y::AbstractVector, c::AbstractVector; hover=nothing)
	wells = []
	for (j, i) in enumerate(unique(sort(c)))
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

function plot_heel_toe(heel_x::AbstractVector, heel_y::AbstractVector, toe_x::AbstractVector, toe_y::AbstractVector, c::AbstractVector; hover=nothing)
	if hover != nothing
		@assert length(hover) == length(heel_x)
	end
	@assert length(heel_x) == length(heel_y)
	@assert length(heel_x) == length(toe_x)
	@assert length(heel_x) == length(toe_y)
	@assert length(heel_x) == length(c)
	traces = []
	for (j, i) in enumerate(unique(sort(c)))
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

function plot_heel_toe(heel_x::AbstractVector, heel_y::AbstractVector, heel_z::AbstractVector, toe_x::AbstractVector, toe_y::AbstractVector, toe_z::AbstractVector, c::AbstractVector; hover=nothing)
	if hover != nothing
		@assert length(hover) == length(heel_x)
	end
	@assert length(heel_x) == length(heel_y)
	@assert length(heel_x) == length(toe_x)
	@assert length(heel_x) == length(toe_y)
	@assert length(heel_x) == length(c)
	traces = []
	for (j, i) in enumerate(unique(sort(c)))
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