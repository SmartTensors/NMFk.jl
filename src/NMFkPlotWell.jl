import Plotly
import PlotlyJS
import Colors
import ColorSchemes

function colorscale(scheme::Symbol; N = 101)
	x = permutedims(0.0:(1.0/(N - 1)):1.0)
	cs = get(ColorSchemes.colorschemes[scheme], x, :clamp)
	cs_rgb = Colors.RGB.(cs)
	return vcat(x, cs_rgb)
  end

function plot_dots(x::AbstractVector, y::AbstractVector, z::AbstractVector; hover=nothing, label=nothing, title::AbstractString="", pointsize=6)
	if hover !== nothing
		@assert length(hover) == length(x)
	end
	if label !== nothing
		@assert length(label) == length(x)
	end
	@assert length(x) == length(y)
	@assert length(x) == length(z)
	l = label === nothing ? Dict(:mode=>"markers") : Dict(:mode=>"markers+text", :text=>label, :textposition=>"left center")
	if eltype(z) <: AbstractFloat
		h = hover === nothing ? Dict() : Dict(:hovertext=>hover, :hoverinfo=>"text")
		p = PlotlyJS.scatter(; x=x, y=y, z=z, l..., marker=Plotly.attr(; size=pointsize, color=z, colorscale=colorscale(:rainbow), colorbar=Plotly.attr(; thickness=20)), h...)
	else
		dots = []
		uz = unique(sort(z))
		if length(uz) > 21
			@warn("Z attribute is potentially not categorical!")
		end
		if length(uz) > 100
			@error("Z attribute is not categorical!")
			throw("Data input error!")
		end
		for (j, i) in enumerate(uz)
			iz = z .== i
			h = hover === nothing ? Dict() : Dict(:hovertext=>hover[iz], :hoverinfo=>"text")
			jj = j % length(NMFk.colors)
			c = jj == 0 ? length(NMFk.colors) : jj
			dots_p = PlotlyJS.scatter(; x=x[iz], y=y[iz], l..., name="$i $(sum(iz))", marker=Plotly.attr(; size=pointsize), color=NMFk.colors[c], h...)
			push!(dots, dots_p)
		end
		p = convert(Array{typeof(dots[1])}, dots)
	end
	return PlotlyJS.plot(p, Plotly.Layout(; title=title, hovermode="closest", yaxis_scaleanchor="x", yaxis_scaleratio=1))

end

function plot_wells(filename::AbstractString, ar...; figuredir::AbstractString=".", title::AbstractString="", plotly=nothing, kw...)
	if plotly === nothing
		p = PlotlyJS.plot(NMFk.plot_wells(ar...; kw...), Plotly.Layout(; title=title, hovermode="closest", yaxis_scaleanchor="x", yaxis_scaleratio=1))
	else
		p = PlotlyJS.plot(plotly, Plotly.Layout(; title=title, hovermode="closest", yaxis_scaleanchor="x", yaxis_scaleratio=1))
		p = Plotly.addtraces(p, NMFk.plot_wells(ar...; kw...)...)
	end
	j = joinpathcheck(figuredir, filename)
	recursivemkdir(j)
	PlotlyJS.savefig(p, j; format="html")
	return p
end

function plot_wells(wx::AbstractVector, wy::AbstractVector; kw...)
	return NMFk.plot_wells(wx, wy, ones(length(wx)); kw...)
end

function plot_wells(wx::AbstractVector, wy::AbstractVector, c::AbstractVector; hover=nothing, label=nothing, pointsize=6)
	if hover !== nothing
		@assert length(hover) == length(wx)
	end
	if label !== nothing
		@assert length(label) == length(wx)
	end
	@assert length(wx) == length(wy)
	@assert length(wx) == length(c)
	wells = []
	for (j, i) in enumerate(unique(sort(c)))
		ic = c .== i
		l = label === nothing ? Dict(:mode=>"markers") : Dict(:mode=>"markers+text", :text=>label, :textposition=>"left center")
		h = hover === nothing ? Dict() : Dict(:hovertext=>hover[ic], :hoverinfo=>"text")
		well_p = PlotlyJS.scatter(;x=wx[ic], y=wy[ic], l..., name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(; size=pointsize), h...)
		push!(wells, well_p)
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_wells(wx::AbstractVector, wy::AbstractVector, wz::AbstractVector, c::AbstractVector; hover=nothing, pointsize=6)
	if hover !== nothing
		@assert length(hover) == length(wx)
	end
	@assert length(wx) == length(wy)
	@assert length(wx) == length(wz)
	@assert length(wx) == length(c)
	wells = []
	for (j, i) in enumerate(unique(sort(c)))
		ic = c .== i
		h = hover === nothing ? Dict() : Dict(:hovertext=>hover[ic], :hoverinfo=>"text")
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
		for k = eachindex(hx)
			well_trace = PlotlyJS.scatter(;x=[hx[k], tx[k]], y=[ty[k], ty[k]], mode="lines+markers", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6), line=Plotly.attr(width=2, color=NMFk.colors[j]), transform=Plotly.attr(type="groupby", groups=fill(i, length(hx)), styles=Plotly.attr(target="$i $(sum(ic))")), color=NMFk.colors[j])
			push!(wells, well_trace)
		end
	end
	return convert(Array{typeof(wells[1])}, wells)
end

function plot_heel_toe(heel_x::AbstractVector, heel_y::AbstractVector, toe_x::AbstractVector, toe_y::AbstractVector, c::AbstractVector; hover=nothing)
	if hover !== nothing
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
		x = vec(hcat([[hx[i] tx[i] NaN] for i = eachindex(hx)]...))
		y = vec(hcat([[hy[i] ty[i] NaN] for i = eachindex(hy)]...))
		if hover !== nothing
			h = vec(hcat([[hover[i] hover[i] NaN] for i = eachindex(hover)]...))
			well_trace = PlotlyJS.scatter(;x=x, y=y, hovertext=h, mode="lines+markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6), line=Plotly.attr(width=2, color=NMFk.colors[j]))
		else
			well_trace = PlotlyJS.scatter(;x=x, y=y, mode="lines+markers", name="$i $(sum(ic))", marker_color=NMFk.colors[j], marker=Plotly.attr(size=6), line=Plotly.attr(width=2, color=NMFk.colors[j]))
		end
		push!(traces, well_trace)
	end
	return convert(Array{typeof(traces[1])}, traces)
end

function plot_heel_toe(heel_x::AbstractVector, heel_y::AbstractVector, heel_z::AbstractVector, toe_x::AbstractVector, toe_y::AbstractVector, toe_z::AbstractVector, c::AbstractVector; hover=nothing)
	if hover !== nothing
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
		x = vec(hcat([[hx[i] tx[i] NaN] for i = eachindex(hx)]...))
		y = vec(hcat([[hy[i] ty[i] NaN] for i = eachindex(hy)]...))
		z = vec(hcat([[hz[i] tz[i] NaN] for i = eachindex(hz)]...))
		if hover !== nothing
			h = vec(hcat([[hover[i] hover[i] NaN] for i = eachindex(hover)]...))
			well_trace = PlotlyJS.scatter3d(;x=x, y=y, z=z, hovertext=h, mode="lines", name="$i $(sum(ic))", marker_color=NMFk.colors[j], line=Plotly.attr(width=6, color=NMFk.colors[j]))
		else
			well_trace = PlotlyJS.scatter3d(;x=x, y=y, z=z, mode="lines", name="$i $(sum(ic))", marker_color=NMFk.colors[j], line=Plotly.attr(width=6, color=NMFk.colors[j]))
		end
		push!(traces, well_trace)
	end
	return convert(Array{typeof(traces[1])}, traces)
end