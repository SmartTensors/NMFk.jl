import DataFrames
import PlotlyJS
import Interpolations
import NearestNeighbors
import Statistics
import ConcaveHull

mapbox_token = "pk.eyJ1IjoibW9udHl2IiwiYSI6ImNsMDhvNTJwMzA1OHgzY256N2c2aDdzdXoifQ.cGUz0Wuc3rYRqGNwm9v5iQ"

regex_lon = r"^[Xx]$|^[Ll]on$|^LONGITUDE$|^LON$|^[Ll]ongitude$" # regex for longitude
regex_lat = r"^[Yy]$|^[Ll]at$|^LATITUDE$|^LAT$|^[Ll]atitude$" # regex for latitude

# Mapbox for a dataframe with multiple columns
function mapbox(df::DataFrames.DataFrame; namesmap=names(df), column::Union{Symbol, AbstractString}="", filename::AbstractString="", title::AbstractString="", title_colorbar::AbstractString=title, title_length::Number=0, categorical::Bool=false, kw...)
	lon, lat = get_lonlat(df)
	if isnothing(lon) || isnothing(lat)
		@error("Longitude and latitude columns are required for plotting!")
		return nothing
	end
	fileroot, fileext = splitext(filename)
	if column == ""
		local col = 1
		for a in names(df)
			if !(occursin(regex_lon, a) || occursin(regex_lat, a))
				varname = namesmap[col]
				col += 1
				println("Plotting $(varname) ...")
				if filename != ""
					aa = replace(string(a), '/' => Char(0x2215))
					f = fileroot * "_" * aa * fileext
				else
					f = ""
				end
				if title_colorbar == ""
					t = plotly_title_length(varname, title_length)
				else
					t = plotly_title_length(title_colorbar, title_length) * "<br>" * plotly_title_length(varname, title_length)
				end
				if categorical
					p = mapbox(lon, lat, string.(df[!, a]); filename=f, title_colorbar=t, title=title, kw...)
				else
					p = mapbox(lon, lat, df[!, a]; filename=f, title_colorbar=t, title=title, kw...)
				end
				display(p)
			else
				if length(namesmap) == length(names(df))
					col += 1
				end
			end
		end
	else
		if filename != ""
			f = fileroot * "_" * column * fileext
		else
			f = ""
		end
		p = mapbox(lon, lat, df[!, column]; filename=f, title_colorbar=plotly_title_length(column, title_length), title=title, kw...)
		display(p)
	end
	return p
end

# Mapbox for a matrix with multiple columns
function mapbox(lon::AbstractVector{T1}, lat::AbstractVector{T1}, M::AbstractMatrix{T2}, names::AbstractVector=["Column $i" for i in axes(M, 2)]; filename::AbstractString="", title::AbstractString="", title_colorbar::AbstractString=title, title_length::Number=0, kw...) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	fileroot, fileext = splitext(filename)
	for i in eachindex(names)
		println("Plotting $(names[i]) ...")
		if filename != ""
			aa = replace(string(names[i]), '/' => '\u2215')
			f = fileroot * "_" * aa * fileext
		else
			f = ""
		end
		if title_colorbar == ""
			t = plotly_title_length(string(names[i]), title_length)
		else
			t = plotly_title_length(title_colorbar, title_length) * "<br>" * plotly_title_length(string(names[i]), title_length)
		end
		p = mapbox(lon, lat, M[:, i]; filename=f, title_colorbar=t, title=title, kw...)
		display(p)
	end
end

function mapbox(
	lon::AbstractVector{T1},
	lat::AbstractVector{T1},
	color::AbstractVector{T2};
	zmin::Number=minimumnan(color),
	zmax::Number=maximumnan(color),
	title::AbstractString="",
	title_colorbar::AbstractString=title,
	title_length::Number=0,
	text::AbstractVector=repeat([""], length(lon)),
	lonc::AbstractFloat=minimumnan(lon) + (maximumnan(lon) - minimumnan(lon)) / 2,
	font_size::Number=14,
	font_size_fig::Number=font_size * 2,
	font_color::AbstractString="black",
	font_color_fig::AbstractString=font_color,
	showlabels::Bool=false,
	label_position::AbstractString="top center",
	label_font_size::Number=font_size,
	label_font_size_fig::Number=font_size_fig,
	label_font_color::AbstractString=font_color,
	label_font_color_fig::AbstractString=font_color_fig,
	line_color::AbstractString="purple",
	line_width::Number=0,
	line_width_fig::Number=line_width * 2,
	marker_color::AbstractString="purple",
	marker_size::Number=0,
	marker_size_fig::Number=marker_size * 2,
	latc::AbstractFloat=minimumnan(lat) + (maximumnan(lat) - minimumnan(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	zoom_fig::Number=zoom,
	dot_size::Number=compute_dot_size(lon, lat, zoom),
	dot_size_fig::Number=dot_size * 2,
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	filename::AbstractString="",
	figuredir::AbstractString=".",
	format::AbstractString=splitext(filename)[end][2:end],
	width::Int=2800,
	height::Int=1400,
	scale::Real=1,
	legend::Bool=true,
	colorbar::Bool=legend,
	traces::AbstractVector=[],
	traces_setup=(; mode="lines", line_width=8, line_color="purple", attributionControl=false),
	colorscale::Symbol=:rainbow,
	paper_bgcolor::AbstractString="#FFF",
	showcount::Bool=true
) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	if title == title_colorbar
		title = ""
	end
	traces = check_traces(traces, traces_setup)
	sort_color = sortpermnan(color)
	if filename != ""
		show_colorbar = true
		for t in traces
			if haskey(t.fields, :line)
				(line_color != "") && (t.fields[:line][:color] = line_color)
				(line_width_fig > 0) && (t.fields[:line][:width] = line_width_fig)
			end
			if haskey(t.fields, :marker)
				(marker_color != "") && (t.fields[:marker][:color] = marker_color)
				(marker_size_fig > 0) && (t.fields[:marker][:size] = marker_size_fig)
			end
			if haskey(t.fields, :showlegend) && t.fields[:showlegend] == true
				show_colorbar = !legend
				if !legend
					t.fields[:showlegend] = false
				end
			end
		end
		if colorbar && show_colorbar
			colorbar_attr = PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(title_colorbar, title_length), titlefont=PlotlyJS.attr(; size=font_size_fig, color=font_color_fig), tickfont=PlotlyJS.attr(; size=font_size_fig, color=font_color_fig))
		else
			colorbar_attr = PlotlyJS.attr()
		end
		marker = PlotlyJS.attr(;
			size=dot_size_fig,
			color=color[sort_color],
			colorscale=NMFk.colorscale(colorscale),
			cmin=zmin,
			cmax=zmax,
			colorbar=colorbar_attr
		)
		plot = PlotlyJS.scattermapbox(;
			lon=lon[sort_color],
			lat=lat[sort_color],
			text=text[sort_color],
			mode=showlabels ? "markers+text" : "markers",
			hoverinfo="text",
			marker=marker,
			textposition=label_position,
			textfont=PlotlyJS.attr(; size=label_font_size_fig, color=label_font_color_fig),
			attributionControl=false,
			showlegend=false
		)
		layout = plotly_layout(lonc, latc, zoom_fig; width=width, height=height, title=title, font_size=font_size_fig, style=style, mapbox_token=mapbox_token)
		p = PlotlyJS.plot([plot, traces...], layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	show_colorbar = true
	for t in traces
		if haskey(t.fields, :line)
			(line_color != "") && (t.fields[:line][:color] = line_color)
			(line_width > 0) && (t.fields[:line][:width] = line_width)
		end
		if haskey(t.fields, :marker)
			(marker_color != "") && (t.fields[:marker][:color] = marker_color)
			(marker_size > 0) && (t.fields[:marker][:size] = marker_size)
		end
		if haskey(t.fields, :showlegend) && t.fields[:showlegend] == true
			show_colorbar = !legend
			if !legend
				t.fields[:showlegend] = false
			end
		end
	end
	if colorbar && show_colorbar
		colorbar_attr = PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(title_colorbar, title_length), titlefont=PlotlyJS.attr(; size=font_size, color=font_color), tickfont=PlotlyJS.attr(; size=font_size, color=font_color))
	else
		colorbar_attr = PlotlyJS.attr()
	end
	marker = PlotlyJS.attr(;
		size=dot_size,
		color=color[sort_color],
		cmin=zmin,
		cmax=zmax,
		colorscale=NMFk.colorscale(colorscale),
		colorbar=colorbar_attr
	)
	plot = PlotlyJS.scattermapbox(;
		lon=lon[sort_color],
		lat=lat[sort_color],
		text=text[sort_color],
		mode=showlabels ? "markers+text" : "markers",
		hoverinfo="text",
		marker=marker,
		textposition=label_position,
		textfont=PlotlyJS.attr(; size=label_font_size, color=label_font_color),
		attributionControl=false,
		showlegend=false
	)
	layout = plotly_layout(lonc, latc, zoom; paper_bgcolor=paper_bgcolor, title=title, font_size=font_size, style=style, mapbox_token=mapbox_token)
	p = PlotlyJS.plot([plot, traces...], layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
	return p
end

function mapbox(
	lon::AbstractVector{T1},
	lat::AbstractVector{T1},
	color::AbstractVector{T2};
	title::AbstractString="",
	title_colorbar::AbstractString="",
	title_length::Number=0,
	text::AbstractVector=repeat([""], length(lon)),
	lonc::AbstractFloat=minimumnan(lon) + (maximumnan(lon) - minimumnan(lon)) / 2,
	font_size::Number=14,
	font_size_fig::Number=font_size * 2,
	font_color::AbstractString="black",
	font_color_fig::AbstractString=font_color,
	showlabels::Bool=false,
	label_position::AbstractString="top center",
	label_font_size::Number=font_size,
	label_font_size_fig::Number=font_size_fig,
	label_font_color::AbstractString=font_color,
	label_font_color_fig::AbstractString=font_color_fig,
	line_color::AbstractString="purple",
	line_width::Number=0,
	line_width_fig::Number=line_width * 2,
	marker_color::AbstractString="purple",
	marker_size::Number=0,
	marker_size_fig::Number=marker_size * 2,
	latc::AbstractFloat=minimumnan(lat) + (maximumnan(lat) - minimumnan(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	zoom_fig::Number=zoom,
	dot_size::Number=compute_dot_size(lon, lat, zoom),
	dot_size_fig::Number=dot_size * 2,
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	filename::AbstractString="",
	figuredir::AbstractString=".",
	format::AbstractString=splitext(filename)[end][2:end],
	width::Int=2800,
	height::Int=1400,
	scale::Real=1,
	legend::Bool=true,
	colorbar::Bool=legend,
	traces::AbstractVector=[],
	traces_setup=(; mode="lines", line_width=8, line_color="purple", attributionControl=false),
	showlegend=false,
	colorscale::Symbol=:rainbow,
	paper_bgcolor::AbstractString="white",
	showcount::Bool=true
) where {T1 <: AbstractFloat, T2 <: Union{Number, Symbol, AbstractString, AbstractChar}}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	if title == title_colorbar
		title = ""
	end
	traces = check_traces(traces, traces_setup)
	if filename != ""
		traces_ = Vector{PlotlyJS.GenericTrace{Dict{Symbol, Any}}}(undef, 0)
		for (j, i) in enumerate(unique(sort(color)))
			iz = color .== i
			jj = j % length(NMFk.colors)
			k = jj == 0 ? length(NMFk.colors) : jj
			marker = PlotlyJS.attr(; size=dot_size_fig, color=NMFk.colors[k], colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(repeat("&nbsp;", title_length) * " colorbar " * title, title_length), titlefont=PlotlyJS.attr(; size=font_size_fig, color=paper_bgcolor), tickfont=PlotlyJS.attr(; size=font_size_fig, color=paper_bgcolor)))
			name = showcount ? "$(string(i)) [$(sum(iz))]" : "$(string(i))"
			t = PlotlyJS.scattermapbox(;
				lon=lon[iz],
				lat=lat[iz],
				text=text[iz],
				hoverinfo="text",
				name=name,
				mode=showlabels ? "markers+text" : "markers",
				marker=marker,
				showlegend=legend,
				textposition=label_position,
				textfont=PlotlyJS.attr(; size=label_font_size_fig, color=label_font_color_fig),
				attributionControl=false)
			push!(traces_, t)
		end
		for t in traces
			if haskey(t.fields, :line)
				(line_color != "") && (t.fields[:line][:color] = line_color)
				(line_width_fig > 0) && (t.fields[:line][:width] = line_width_fig)
			end
			if haskey(t.fields, :marker)
				(marker_color != "") && (t.fields[:marker][:color] = marker_color)
				(marker_size_fig > 0) && (t.fields[:marker][:size] = marker_size_fig)
			end
			push!(traces_, t)
		end
		layout = plotly_layout(lonc, latc, zoom_fig; paper_bgcolor=paper_bgcolor, font_size=font_size_fig, font_color=font_color_fig, title=title, style=style, mapbox_token=mapbox_token)
		p = PlotlyJS.plot(traces_, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	traces_ = Vector{PlotlyJS.GenericTrace{Dict{Symbol, Any}}}(undef, 0)
	for (j, i) in enumerate(unique(sort(color)))
		iz = color .== i
		jj = j % length(NMFk.colors)
		k = jj == 0 ? length(NMFk.colors) : jj
		marker = PlotlyJS.attr(; size=dot_size, color=NMFk.colors[k], colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(repeat("&nbsp;", title_length) * " colorbar " * title, title_length), titlefont=PlotlyJS.attr(; size=font_size, color=paper_bgcolor), tickfont=PlotlyJS.attr(; size=font_size, color=paper_bgcolor)))
		name = showcount ? "$(string(i)) [$(sum(iz))]" : "$(string(i))"
		t = PlotlyJS.scattermapbox(;
			lon=lon[iz],
			lat=lat[iz],
			text=text[iz],
			hoverinfo="text",
			name=name,
			mode=showlabels ? "markers+text" : "markers",
			marker=marker,
			showlegend=true,
			textposition=label_position,
			textfont=PlotlyJS.attr(; size=label_font_size, color=label_font_color),
			attributionControl=false)
		push!(traces_, t)
	end
	for t in traces
		if haskey(t.fields, :line)
			(line_color != "") && (t.fields[:line][:color] = line_color)
			(line_width > 0) && (t.fields[:line][:width] = line_width)
		end
		if haskey(t.fields, :marker)
			(marker_color != "") && (t.fields[:marker][:color] = marker_color)
			(marker_size > 0) && (t.fields[:marker][:size] = marker_size)
		end
		push!(traces_, t)
	end
	layout = plotly_layout(lonc, latc, zoom; paper_bgcolor=paper_bgcolor, title=title, font_size=font_size, font_color=font_color, style=style, mapbox_token=mapbox_token)
	p = PlotlyJS.plot(traces_, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
	display(p)
	return p
end

function mapbox(lon::AbstractFloat=-105.9378, lat::AbstractFloat=35.6870; color::AbstractString="purple", text::AbstractString="EnviTrace LLC", dot_size::Number=12, kw...)
	return mapbox([lon], [lat], [color]; text=[text], dot_size=dot_size, legend=false, kw...)
end

function check_traces(traces::AbstractVector, traces_setup::NamedTuple)
	if length(traces) == 0
		traces = Vector{PlotlyJS.GenericTrace{Dict{Symbol, Any}}}(undef, 0)
	elseif !(eltype(traces) <: PlotlyJS.GenericTrace{Dict{Symbol, Any}})
		traces_vector = Vector{PlotlyJS.GenericTrace{Dict{Symbol, Any}}}(undef, length(traces))
		for (i, t) in enumerate(traces)
			if haskey(t, :lat) && haskey(t, :lon)
				name = haskey(t, :name) ? t[:name] : "Domain"
				if length(t[:lon]) != length(t[:lat])
					@error("The length of lon and lat must be the same!")
					return nothing
				elseif length(t[:lon]) > 2
					t_new = (; lon=[t[:lon][1:2]; t[:lon]], lat=[t[:lat][1:2]; t[:lat]])
					traces_vector[i] = PlotlyJS.scattermapbox(; traces_setup..., name=name, t_new...)
				else
					traces_vector[i] = PlotlyJS.scattermapbox(; traces_setup..., t...)
				end
			else
				traces_vector[i] = PlotlyJS.scattermapbox(; traces_setup...)
			end
		end
		traces = traces_vector
	end
	return traces
end

# Plotly map layout
function plotly_layout(lonc::Number=-105.9378, latc::Number=35.6870, zoom::Number=4; paper_bgcolor::AbstractString="white", width::Int=2800, height::Int=1400, title::AbstractString="", font_size::Number=14, font_color="black", style="mapbox://styles/mapbox/satellite-streets-v12", mapbox_token=NMFk.mapbox_token)
	layout = PlotlyJS.Layout(;
		margin=PlotlyJS.attr(; r=0, t=0, b=0, l=0),
		autosize=true,
		width=width,
		height=height,
		legend=PlotlyJS.attr(; title_text=title, title_font_size=font_size, itemsizing="constant", font=PlotlyJS.attr(; size=font_size, color=font_color), bgcolor=paper_bgcolor),
		paper_bgcolor=paper_bgcolor,
		mapbox=PlotlyJS.attr(;
			accesstoken=mapbox_token,
			style=style,
			center=PlotlyJS.attr(; lon=lonc, lat=latc),
			zoom=zoom
		)
	)
	return layout
end

function plotly_title_length(title::AbstractString, length::Number)
	if length <= 0
		return title
	else
		title_vector = split(title, ' ')
		pushfirst!(title_vector, repeat("&nbsp;", length)) # adding nonbreaking spaces to control the colorbar size/position
		title_adjusted = join(title_vector, "<br>")
		return title_adjusted
	end
end

function compute_zoom_dot_size(x::AbstractVector, y::AbstractVector)
	zoom = compute_zoom(x, y)
	dot_size = compute_dot_size(x, y, zoom)
	return zoom, dot_size
end

function compute_zoom(x::AbstractVector, y::AbstractVector)
	coordmask = .!isnan.(x) .| .!isnan.(y)
	lonmin = minimumnan(x[coordmask])
	lonmax = maximumnan(x[coordmask])
	latmin = minimumnan(y[coordmask])
	latmax = maximumnan(y[coordmask])
	lonr = lonmax - lonmin
	latr = latmax - latmin
	coord_range = max(lonr, latr)
	dx_range = [0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096, 0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568, 47.5136, 98.304, 190.0544, 360.0]
	zoom_range = 19:-1:0
	zoom_itp = Interpolations.interpolate((dx_range,), zoom_range, Interpolations.Gridded(Interpolations.Linear()))
	zoom = zoom_itp[coord_range]
	return zoom
end

function compute_dot_size(x::AbstractVector, y::AbstractVector, zoom::Number)
	coordmask = .!isnan.(x) .| .!isnan.(y)
	coord = unique([x[coordmask]'; y[coordmask]']; dims=2)
	kd = NearestNeighbors.KDTree(coord)
	d = [i[2] for i in NearestNeighbors.knn(kd, coord, 2, true)[2]]
	d_metric = Statistics.mean(d)
	dot_size = 3 + Int(ceil(d_metric * zoom * zoom * zoom)) / 2
	# @show d_metric, dot_size, zoom
	return dot_size
end

# -- GeoJSON helpers ---------------------------------------------------------
# Tile-building logic mirrors SmartBack's geomltools_service.jl grid masking
# so the interpolated field can be emitted as square GeoJSON features for Mapbox.
function _grid_edges(vec::AbstractVector{T}) where {T <: Real}
	n = length(vec)
	if n == 0
		return Float64[]
	elseif n == 1
		step = one(eltype(vec))
		return [Float64(vec[1] - step / 2), Float64(vec[1] + step / 2)]
	end
	edges = Vector{Float64}(undef, n + 1)
	edges[2:n] = (vec[1:n-1] .+ vec[2:n]) ./ 2
	first_step = vec[2] - vec[1]
	last_step = vec[end] - vec[end-1]
	edges[1] = Float64(vec[1] - first_step / 2)
	edges[end] = Float64(vec[end] + last_step / 2)
	return edges
end

function _point_in_polygon(pt::Tuple{Float64, Float64}, polygon::Vector{Tuple{Float64, Float64}})
	x, y = pt
	inside = false
	n = length(polygon)
	if n < 3
		return false
	end
	x1, y1 = polygon[1]
	for i = 1:n
		x2, y2 = polygon[i]
		if ((y1 > y) != (y2 > y)) && (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
			inside = !inside
		end
		x1, y1 = x2, y2
	end
	return inside
end

function _compute_concave_hull_vertices(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real})
	if length(lon) < 3
		return nothing
	end
	pts = [[Float64(lon[i]), Float64(lat[i])] for i in eachindex(lon)]
	try
		hull = ConcaveHull.concave_hull(pts)
		verts = [(Float64(p[1]), Float64(p[2])) for p in hull.vertices]
		if isempty(verts)
			return nothing
		end
		if verts[1] != verts[end]
			push!(verts, verts[1])
		end
		return verts
	catch e
		@warn "Concave hull computation failed; falling back to bounding box" error=e
		return nothing
	end
end

function _expand_polygon_vertices(
	vertices::Vector{Tuple{Float64, Float64}};
	lon_span::Real,
	lat_span::Real,
	padding_fraction::Real,
	margin::Real
)
	if padding_fraction <= 0 && margin <= 0
		return vertices
	end
	if length(vertices) < 3
		return vertices
	end
	pad_lon = max(lon_span, 1e-6) * max(padding_fraction, 0.0)
	pad_lat = max(lat_span, 1e-6) * max(padding_fraction, 0.0)
	closed = vertices[1] == vertices[end]
	pts = closed ? vertices[1:end-1] : vertices
	cx = Statistics.mean(first.(pts))
	cy = Statistics.mean(last.(pts))
	expanded = Vector{Tuple{Float64, Float64}}(undef, length(pts))
	for (idx, (vx, vy)) in enumerate(pts)
		vec_lon = vx - cx
		vec_lat = vy - cy
		norm = sqrt(vec_lon^2 + vec_lat^2)
		if norm < 1e-9
			# Rare degenerate case where a vertex sits at the centroid; nudge eastward
			dir_lon, dir_lat = 1.0, 0.0
		else
			dir_lon = vec_lon / norm
			dir_lat = vec_lat / norm
		end
		pad_component = sqrt((pad_lon * abs(dir_lon))^2 + (pad_lat * abs(dir_lat))^2)
		delta = pad_component + max(margin, 0.0)
		expanded[idx] = (vx + dir_lon * delta, vy + dir_lat * delta)
	end
	if closed
		push!(expanded, expanded[1])
	end
	return expanded
end

function _build_geojson_tiles(lon_grid::AbstractVector{<:Real}, lat_grid::AbstractVector{<:Real}, values::AbstractMatrix{<:Real}; mask_polygon::Union{Nothing, Vector{Tuple{Float64, Float64}}}=nothing)
	if length(lon_grid) < 2 || length(lat_grid) < 2
		return (Dict("type" => "FeatureCollection", "features" => Any[]), String[], Float64[])
	end
	lon_edges = _grid_edges(lon_grid)
	lat_edges = _grid_edges(lat_grid)
	features = Vector{Dict{String, Any}}()
	feature_ids = String[]
	tile_values = Float64[]
	rows, cols = size(values)
	max_i = min(rows - 1, length(lat_edges) - 1)
	max_j = min(cols - 1, length(lon_edges) - 1)
	for i = 1:max_i
		for j = 1:max_j
			val = Float64(values[i, j])
			if !isfinite(val)
				continue
			end
			if mask_polygon !== nothing
				center = ((lon_edges[j] + lon_edges[j + 1]) / 2, (lat_edges[i] + lat_edges[i + 1]) / 2)
				if !_point_in_polygon(center, mask_polygon)
					continue
				end
			end
			fid = string("tile_", i, "_", j)
			coords = [
				[lon_edges[j], lat_edges[i]],
				[lon_edges[j + 1], lat_edges[i]],
				[lon_edges[j + 1], lat_edges[i + 1]],
				[lon_edges[j], lat_edges[i + 1]],
				[lon_edges[j], lat_edges[i]]
			]
			feature = Dict(
				"type" => "Feature",
				"id" => fid,
				"geometry" => Dict("type" => "Polygon", "coordinates" => [coords]),
				"properties" => Dict("value" => val)
			)
			push!(features, feature)
			push!(feature_ids, fid)
			push!(tile_values, val)
		end
	end
	geojson = Dict("type" => "FeatureCollection", "features" => features)
	return geojson, feature_ids, tile_values
end

"""
mapbox_contour(lon, lat, values; resolution=50, power=2, smoothing=0.0, filename="", kw...)

Create GeoJSON-based continuous contour heatmap using IDW (Inverse Distance Weighting) interpolation.

# Arguments
- `lon::AbstractVector`: Vector of longitude coordinates
- `lat::AbstractVector`: Vector of latitude coordinates
- `values::AbstractVector`: Vector of values to interpolate
- `resolution::Int=50`: Grid resolution for interpolation (higher = smoother but slower)
- `power::Real=2`: IDW power parameter (higher = more localized interpolation)
- `smoothing::Real=0.0`: Smoothing parameter for interpolation
- `contour_levels::Int=10`: Number of contour levels
- `filename::AbstractString=""`: Output filename for saving the plot
- `title::AbstractString=""`: Plot title
- `title_colorbar::AbstractString=title`: Colorbar title
- `colorscale::Symbol=:viridis`: Color scale for the heatmap
- `opacity::Real=0.7`: Opacity of the contour layer
- `show_points::Bool=false`: Whether to show original data points
- `concave_hull::Bool=true`: If true, derive extent/masking from a ConcaveHull envelope
- `hull_padding::Real=0.02`: Fractional padding applied to the concave hull shape itself
- `extra_margin::Real=0.0`: Absolute degree margin added radially outside the hull
- `show_locations::Bool=true`: Display input locations as colored circular markers
- `location_color::AbstractString="purple"`: Marker color used for the location circles
- `location_size::Number=10`: Marker diameter for the location circles
- `location_names::Union{Nothing, AbstractVector}=nothing`: Optional labels plotted next to each location
- `kw...`: Additional keyword arguments passed to the mapbox function

# Example
```julia
lon = [-105.0, -104.5, -104.0, -103.5]
lat = [35.5, 36.0, 36.5, 37.0]
values = [10.0, 15.0, 20.0, 25.0]
p = mapbox_contour(lon, lat, values; resolution=100, power=2, filename="contour_map.html")
```
"""
function mapbox_contour(
	lon::AbstractVector{T1},
	lat::AbstractVector{T1},
	values::AbstractVector{T2};
	resolution::Int=50,
	power::Real=2,
	smoothing::Real=0.0,
	contour_levels::Int=10,
	filename::AbstractString="",
	title::AbstractString="",
	title_colorbar::AbstractString=title,
	colorscale::Symbol=:viridis,
	opacity::Real=0.7,
	show_locations::Bool=true,
	location_color::AbstractString="purple",
	location_size::Number=10,
	location_names::AbstractVector=String[],
	lonc::Real=minimumnan(lon) + (maximumnan(lon) - minimumnan(lon)) / 2,
	latc::Real=minimumnan(lat) + (maximumnan(lat) - minimumnan(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	figuredir::AbstractString=".",
	format::AbstractString=filename == "" ? "html" : splitext(filename)[end][2:end],
	dpi::Int=200,
	width::Int=dpi * 14,
	height::Int=dpi * 7,
	scale::Real=1,
	font_size::Number=14,
	concave_hull::Bool=true,
	hull_padding::Real=0.02,
	extra_margin::Real=0.005,
	quiet::Bool=false,
	kw...
) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	@assert length(lon) == length(lat) == length(values)

	# Remove NaN values
	valid_mask = .!isnan.(lon) .& .!isnan.(lat) .& .!isnan.(values)
	lon_clean = lon[valid_mask]
	lat_clean = lat[valid_mask]
	values_clean = values[valid_mask]
	names_clean = nothing
	if !isempty(location_names)
		try
			name_vec = collect(location_names)
			if length(name_vec) == length(lon)
				names_clean = name_vec[valid_mask]
			elseif length(name_vec) == length(lon_clean)
				names_clean = name_vec
			else
				@warn "location_names length does not match lon/lat; skipping labels"
			end
		catch e
			@warn "Failed to process location_names; skipping labels" error=e
		end
	end
	if names_clean !== nothing
		names_clean = string.(names_clean)
	end

	if length(lon_clean) < 3
		@error("At least 3 valid data points are required for interpolation!")
		return nothing
	end

	hull_vertices = nothing
	if concave_hull
		hull_vertices = _compute_concave_hull_vertices(lon_clean, lat_clean)
		if hull_vertices === nothing || length(hull_vertices) < 3
			@info "Concave hull unavailable; reverting to padded bounding box"
			hull_vertices = nothing
		end
	end

	lon_source_raw = hull_vertices === nothing ? lon_clean : first.(hull_vertices)
	lat_source_raw = hull_vertices === nothing ? lat_clean : last.(hull_vertices)
	lon_range_raw = maximum(lon_source_raw) - minimum(lon_source_raw)
	lat_range_raw = maximum(lat_source_raw) - minimum(lat_source_raw)
	effective_margin = max(0.0, extra_margin)
	effective_padding = max(0.0, hull_padding)
	if hull_vertices !== nothing && (effective_padding > 0 || effective_margin > 0)
		hull_vertices = _expand_polygon_vertices(
			hull_vertices;
			lon_span=lon_range_raw,
			lat_span=lat_range_raw,
			padding_fraction=effective_padding,
			margin=effective_margin
		)
		lon_source_raw = first.(hull_vertices)
		lat_source_raw = last.(hull_vertices)
		lon_range_raw = maximum(lon_source_raw) - minimum(lon_source_raw)
		lat_range_raw = maximum(lat_source_raw) - minimum(lat_source_raw)
	end

	lon_source = lon_source_raw
	lat_source = lat_source_raw
	lon_range = lon_range_raw
	lat_range = lat_range_raw
	lon_span = lon_range == 0 ? 1e-6 : lon_range
	lat_span = lat_range == 0 ? 1e-6 : lat_range
	padding = hull_vertices === nothing ? 0.1 : 0.0
	margin = hull_vertices === nothing ? effective_margin : 0.0
	lon_min = minimum(lon_source) - lon_span * padding - margin
	lon_max = maximum(lon_source) + lon_span * padding + margin
	lat_min = minimum(lat_source) - lat_span * padding - margin
	lat_max = maximum(lat_source) + lat_span * padding + margin

	# Create grid
	lon_grid = range(lon_min, lon_max, length=resolution)
	lat_grid = range(lat_min, lat_max, length=resolution)

	# Perform IDW interpolation
	z_grid = Matrix{Float64}(undef, resolution, resolution)
	for (i, lat_interp) in enumerate(lat_grid)
		for (j, lon_interp) in enumerate(lon_grid)
			# Calculate distances to all data points
			distances = sqrt.((lon_clean .- lon_interp).^2 .+ (lat_clean .- lat_interp).^2)
			min_dist = minimum(distances)
			if min_dist < 1e-10 # Handle case where interpolation point coincides with data point
				closest_idx = findfirst(distances .== min_dist)
				z_grid[i, j] = values_clean[closest_idx]
			else
				weights = 1.0 ./ (distances.^power .+ smoothing)
				z_grid[i, j] = sum(weights .* values_clean) / sum(weights)
			end
		end
	end

	geojson_tiles, geojson_ids, geojson_values = _build_geojson_tiles(lon_grid, lat_grid, z_grid; mask_polygon=hull_vertices)
	traces = PlotlyJS.GenericTrace{Dict{Symbol, Any}}[]
	if !isempty(geojson_ids)
		min_val = minimum(geojson_values)
		max_val = maximum(geojson_values)
		plot_values = copy(geojson_values)
		if contour_levels > 1 && max_val > min_val
			step = (max_val - min_val) / contour_levels
			if step > 0
				for idx in eachindex(plot_values)
					plot_values[idx] = min_val + round((geojson_values[idx] - min_val) / step) * step
				end
			end
		end
		zmin = min_val
		zmax = max_val == min_val ? min_val + 1e-9 : max_val
		choropleth_trace = PlotlyJS.choroplethmapbox(
			geojson=geojson_tiles,
			locations=geojson_ids,
			z=plot_values,
			customdata=geojson_values,
			featureidkey="id",
			colorscale=NMFk.colorscale(colorscale),
			opacity=opacity,
			zmin=zmin,
			zmax=zmax,
			marker=PlotlyJS.attr(line=PlotlyJS.attr(width=0)),
			colorbar=PlotlyJS.attr(
				thickness=30, len=0.5,
				title=title_colorbar,
				titlefont=PlotlyJS.attr(size=font_size),
				tickfont=PlotlyJS.attr(size=font_size)
			),
			hovertemplate="<b>Value:</b> %{customdata:.3f}<extra></extra>",
			name="Interpolated Field",
			showscale=true,
			showlegend=false
		)
		push!(traces, choropleth_trace)
	else
		fallback_lon = Float64[]
		fallback_lat = Float64[]
		fallback_vals = Float64[]
		for i in 1:resolution
			for j in 1:resolution
				val = z_grid[i, j]
				if isfinite(val)
					push!(fallback_lon, lon_grid[j])
					push!(fallback_lat, lat_grid[i])
					push!(fallback_vals, val)
				end
			end
		end
		if !isempty(fallback_lon)
			contour_trace = PlotlyJS.scattermapbox(
				lon=fallback_lon,
				lat=fallback_lat,
				mode="markers",
				marker=PlotlyJS.attr(
					size=10,
					color=fallback_vals,
					colorscale=NMFk.colorscale(colorscale),
					opacity=opacity,
					colorbar=PlotlyJS.attr(
						thickness=30, len=0.5,
						title=title_colorbar,
						titlefont=PlotlyJS.attr(size=font_size),
						tickfont=PlotlyJS.attr(size=font_size)
					)
				),
				hovertemplate="<b>Lon:</b> %{lon}<br><b>Lat:</b> %{lat}<br><b>Value:</b> %{marker.color}<extra></extra>",
				showlegend=false,
				name="Interpolated Surface"
			)
			push!(traces, contour_trace)
		end
	end

	# Add location markers if requested
	if show_locations && !isempty(lon_clean)
		marker_attr = PlotlyJS.attr(
			size=location_size,
			color=location_color,
			opacity=0.9,
			line=PlotlyJS.attr(color=location_color, width=0)
		)
		if names_clean === nothing
			location_trace = PlotlyJS.scattermapbox(
				lon=lon_clean,
				lat=lat_clean,
				mode="markers",
				marker=marker_attr,
				name="Locations",
				hovertemplate="<b>Lon:</b> %{lon:.4f}<br><b>Lat:</b> %{lat:.4f}<extra></extra>",
				showlegend=false
			)
		else
			location_trace = PlotlyJS.scattermapbox(
				lon=lon_clean,
				lat=lat_clean,
				mode="markers+text",
				marker=marker_attr,
				text=names_clean,
				textposition="top center",
				textfont=PlotlyJS.attr(color=location_color, size=max(8, Int(round(font_size - 2)))),
				name="Locations",
				hovertemplate="<b>%{text}</b><br>Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<extra></extra>",
				showlegend=false
			)
		end
		push!(traces, location_trace)
	end

	# Create layout
	layout = plotly_layout(
		lonc, latc, zoom;
		width=width,
		height=height,
		title=title,
		font_size=font_size,
		style=style,
		mapbox_token=mapbox_token
	)

	# Create plot
	p = PlotlyJS.plot(traces, layout; config=PlotlyJS.PlotConfig(scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
	# Display plot
	!quiet && display(p)

	# Save if filename provided
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end

	return p
end

"""
	idw_interpolate(x_data, y_data, values, x_interp, y_interp; power=2, smoothing=0.0)

Perform Inverse Distance Weighting (IDW) interpolation for a single point.

# Arguments
- `x_data::AbstractVector`: X coordinates of data points
- `y_data::AbstractVector`: Y coordinates of data points
- `values::AbstractVector`: Values at data points
- `x_interp::Real`: X coordinate for interpolation
- `y_interp::Real`: Y coordinate for interpolation
- `power::Real=2`: IDW power parameter
- `smoothing::Real=0.0`: Smoothing parameter

# Returns
- Interpolated value at (x_interp, y_interp)
"""
function idw_interpolate(
	x_data::AbstractVector,
	y_data::AbstractVector,
	values::AbstractVector,
	x_interp::Real,
	y_interp::Real;
	power::Real=2,
	smoothing::Real=0.0
)
	distances = sqrt.((x_data .- x_interp).^2 .+ (y_data .- y_interp).^2)

	# Handle case where interpolation point coincides with data point
	min_dist = minimumnan(distances)
	if min_dist < 1e-10
		closest_idx = findfirst(distances .== min_dist)
		return values[closest_idx]
	end

	# IDW interpolation
	weights = 1.0 ./ (distances.^power .+ smoothing)
	return sum(weights .* values) / sum(weights)
end

"""
	mapbox_contour(df, column; kw...)

Create GeoJSON-based continuous contour heatmap from a DataFrame using IDW interpolation.

# Arguments
- `df::DataFrames.DataFrame`: DataFrame containing longitude, latitude, and value columns
- `column::Union{Symbol, AbstractString}`: Column name containing values to interpolate
- `kw...`: Additional keyword arguments passed to mapbox_contour

# Returns
- PlotlyJS plot object with contour heatmap overlay

# Example
```julia
df = DataFrames.DataFrame(
	lon=[-105.0, -104.5, -104.0, -103.5],
	lat=[35.5, 36.0, 36.5, 37.0],
	temperature=[10.0, 15.0, 20.0, 25.0]
)
p = mapbox_contour(df, :temperature; resolution=100, filename="temp_contour.html")
```
"""
function mapbox_contour(df::DataFrames.DataFrame, column::Union{Symbol, AbstractString}; kw...)
	lon, lat = get_lonlat(df)
	if isnothing(lon) || isnothing(lat)
		@error("Longitude and latitude columns are required for plotting!")
		return nothing
	end

	if !(column in names(df))
		@error("Column '$column' not found in DataFrame!")
		return nothing
	end

	return mapbox_contour(lon, lat, df[!, column]; kw...)
end