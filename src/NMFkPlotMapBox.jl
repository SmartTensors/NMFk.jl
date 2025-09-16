import DataFrames
import PlotlyJS
import Interpolations
import NearestNeighbors
import Statistics

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
				println("Ploting $(varname) ...")
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
function mapbox(lon::AbstractVector{T1}, lat::AbstractVector{T1}, M::AbstractMatrix{T2}, names::AbstractVector=["Column $i" for i in eachcol(M)]; filename::AbstractString="", title::AbstractString="", title_colorbar::AbstractString=title, title_length::Number=0, kw...) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	fileroot, fileext = splitext(filename)
	for i in eachindex(names)
		println("Ploting $(names[i]) ...")
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
	lonc::AbstractFloat=minimum(lon) + (maximum(lon) - minimum(lon)) / 2,
	font_size::Number=14,
	font_size_fig::Number=font_size * 2,
	font_color::AbstractString="black",
	font_color_fig::AbstractString=font_color,
	line_color::AbstractString="purple",
	line_width::Number=0,
	line_width_fig::Number=line_width * 2,
	marker_color::AbstractString="purple",
	marker_size::Number=0,
	marker_size_fig::Number=marker_size * 2,
	latc::AbstractFloat=minimum(lat) + (maximum(lat) - minimum(lat)) / 2,
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
			mode="markers",
			hoverinfo="text",
			marker=marker,
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
		mode="markers",
		hoverinfo="text",
		marker=marker,
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
	lonc::AbstractFloat=minimum(lon) + (maximum(lon) - minimum(lon)) / 2,
	font_size::Number=14,
	font_size_fig::Number=font_size * 2,
	font_color::AbstractString="black",
	font_color_fig::AbstractString=font_color,
	line_color::AbstractString="purple",
	line_width::Number=0,
	line_width_fig::Number=line_width * 2,
	marker_color::AbstractString="purple",
	marker_size::Number=0,
	marker_size_fig::Number=marker_size * 2,
	latc::AbstractFloat=minimum(lat) + (maximum(lat) - minimum(lat)) / 2,
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
				mode="markers",
				marker=marker,
				showlegend=legend,
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
			mode="markers",
			marker=marker,
			showlegend=true,
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
	lonmin = minimum(x[coordmask])
	lonmax = maximum(x[coordmask])
	latmin = minimum(y[coordmask])
	latmax = maximum(y[coordmask])
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
- `kw...`: Additional keyword arguments passed to the mapbox function

# Returns
- PlotlyJS plot object with contour heatmap overlay

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
	show_points::Bool=false,
	point_size::Number=5,
	lonc::Real=minimum(lon) + (maximum(lon) - minimum(lon)) / 2,
	latc::Real=minimum(lat) + (maximum(lat) - minimum(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	figuredir::AbstractString=".",
	format::AbstractString=filename == "" ? "html" : splitext(filename)[end][2:end],
	width::Int=2800,
	height::Int=1400,
	scale::Real=1,
	font_size::Number=14,
	kw...
) where {T1 <: AbstractFloat, T2 <: AbstractFloat}

	@assert length(lon) == length(lat) == length(values)

	# Remove NaN values
	valid_mask = .!isnan.(lon) .& .!isnan.(lat) .& .!isnan.(values)
	lon_clean = lon[valid_mask]
	lat_clean = lat[valid_mask]
	values_clean = values[valid_mask]

	if length(lon_clean) < 3
		@error("At least 3 valid data points are required for interpolation!")
		return nothing
	end

	# Create interpolation grid
	lon_range = maximum(lon_clean) - minimum(lon_clean)
	lat_range = maximum(lat_clean) - minimum(lat_clean)

	# Add padding to grid bounds
	padding = 0.1
	lon_min = minimum(lon_clean) - lon_range * padding
	lon_max = maximum(lon_clean) + lon_range * padding
	lat_min = minimum(lat_clean) - lat_range * padding
	lat_max = maximum(lat_clean) + lat_range * padding

	# Create grid
	lon_grid = range(lon_min, lon_max, length=resolution)
	lat_grid = range(lat_min, lat_max, length=resolution)

	# Perform IDW interpolation
	z_grid = Matrix{Float64}(undef, resolution, resolution)

	for (i, lat_interp) in enumerate(lat_grid)
		for (j, lon_interp) in enumerate(lon_grid)
			# Calculate distances to all data points
			distances = sqrt.((lon_clean .- lon_interp).^2 + (lat_clean .- lat_interp).^2)

			# Handle case where interpolation point coincides with data point
			min_dist = minimum(distances)
			if min_dist < 1e-10
				closest_idx = findfirst(distances .== min_dist)
				z_grid[i, j] = values_clean[closest_idx]
			else
				# IDW interpolation
				weights = 1.0 ./ (distances.^power .+ smoothing)
				z_grid[i, j] = sum(weights .* values_clean) / sum(weights)
			end
		end
	end

	# Create dense grid of interpolated points for smooth heatmap effect
	grid_lon = Float64[]
	grid_lat = Float64[]
	grid_values = Float64[]

	# Use a denser grid for better coverage - reduce subsampling
	subsample = max(1, resolution ÷ 10)  # Less aggressive subsampling for better coverage
	lat_indices = 1:subsample:resolution
	lon_indices = 1:subsample:resolution

	for (ii, i) in enumerate(lat_indices)
		for (jj, j) in enumerate(lon_indices)
			push!(grid_lon, lon_grid[j])
			push!(grid_lat, lat_grid[i])
			push!(grid_values, z_grid[i, j])
		end
	end

	# Calculate appropriate marker size for good coverage
	# Base size on the grid spacing and zoom level
	grid_spacing_lon = (lon_max - lon_min) / length(lon_indices)
	grid_spacing_lat = (lat_max - lat_min) / length(lat_indices)
	avg_spacing = (grid_spacing_lon + grid_spacing_lat) / 2

	# Make markers large enough to overlap and create continuous appearance
	marker_size = max(15, Int(round(avg_spacing * zoom * zoom * 100)))

	# Alternative approach: Create multiple traces for better coverage
	traces = []

	# Create main heatmap using densitymapbox for better continuous appearance
	if length(grid_lon) > 0
		# Create density heatmap trace
		density_trace = PlotlyJS.densitymapbox(
			lon=repeat(grid_lon, 3),  # Repeat points for better density
			lat=repeat(grid_lat, 3),
			z=repeat(grid_values, 3),
			colorscale=NMFk.colorscale(colorscale),
			opacity=opacity,
			radius=max(5, Int(round(marker_size / 2))),  # Radius in pixels
			colorbar=PlotlyJS.attr(
				title=title_colorbar,
				titlefont=PlotlyJS.attr(size=font_size),
				tickfont=PlotlyJS.attr(size=font_size)
			),
			hovertemplate="<b>Lon:</b> %{lon}<br><b>Lat:</b> %{lat}<br><b>Value:</b> %{z}<extra></extra>",
			showlegend=false,
			name="Heatmap"
		)
		push!(traces, density_trace)
	else
		# Fallback to scattermapbox if densitymapbox fails
		contour_trace = PlotlyJS.scattermapbox(
			lon=grid_lon,
			lat=grid_lat,
			mode="markers",
			marker=PlotlyJS.attr(
				size=marker_size,
				color=grid_values,
				colorscale=NMFk.colorscale(colorscale),
				opacity=opacity,
				colorbar=PlotlyJS.attr(
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

	# Add data points if requested
	if show_points
		points_trace = PlotlyJS.scattermapbox(
			lon=lon_clean,
			lat=lat_clean,
			text=["Lon: $(round(lon_clean[i], digits=4))<br>Lat: $(round(lat_clean[i], digits=4))<br>Value: $(round(values_clean[i], digits=2))" for i in eachindex(lon_clean)],
			mode="markers",
			marker=PlotlyJS.attr(
				size=point_size,
				color="white",
				line=PlotlyJS.attr(color="black", width=1)
			),
			name="Data Points",
			hoverinfo="text",
			showlegend=true
		)
		push!(traces, points_trace)
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

	# Save if filename provided
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end

	return p
end

"""
	mapbox_contour_simple(lon, lat, values; resolution=30, kw...)

Simple version of mapbox_contour that creates a visible heatmap using overlapping circular markers.
This is a fallback when the density-based approach doesn't work well.

# Arguments
- `lon::AbstractVector`: Vector of longitude coordinates
- `lat::AbstractVector`: Vector of latitude coordinates
- `values::AbstractVector`: Vector of values to interpolate
- `resolution::Int=30`: Grid resolution for interpolation
- `marker_scale::Real=2.0`: Scaling factor for marker size
- `kw...`: Additional keyword arguments

# Returns
- PlotlyJS plot object with simple heatmap overlay
"""
function mapbox_contour_simple(
	lon::AbstractVector{T1},
	lat::AbstractVector{T1},
	values::AbstractVector{T2};
	resolution::Int=30,
	power::Real=2,
	marker_scale::Real=2.0,
	title::AbstractString="",
	title_colorbar::AbstractString=title,
	colorscale::Symbol=:viridis,
	opacity::Real=0.8,
	show_points::Bool=false,
	point_size::Number=8,
	lonc::Real=minimum(lon) + (maximum(lon) - minimum(lon)) / 2,
	latc::Real=minimum(lat) + (maximum(lat) - minimum(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	filename::AbstractString="",
	figuredir::AbstractString=".",
	format::AbstractString=filename == "" ? "html" : splitext(filename)[end][2:end],
	width::Int=2800,
	height::Int=1400,
	scale::Real=1,
	font_size::Number=14,
	kw...
) where {T1 <: AbstractFloat, T2 <: AbstractFloat}

	@assert length(lon) == length(lat) == length(values)

	# Remove NaN values
	valid_mask = .!isnan.(lon) .& .!isnan.(lat) .& .!isnan.(values)
	lon_clean = lon[valid_mask]
	lat_clean = lat[valid_mask]
	values_clean = values[valid_mask]

	if length(lon_clean) < 3
		@error("At least 3 valid data points are required for interpolation!")
		return nothing
	end

	# Create interpolation grid
	lon_range = maximum(lon_clean) - minimum(lon_clean)
	lat_range = maximum(lat_clean) - minimum(lat_clean)

	# Add padding to grid bounds
	padding = 0.05
	lon_min = minimum(lon_clean) - lon_range * padding
	lon_max = maximum(lon_clean) + lon_range * padding
	lat_min = minimum(lat_clean) - lat_range * padding
	lat_max = maximum(lat_clean) + lat_range * padding

	# Create grid
	lon_grid = range(lon_min, lon_max, length=resolution)
	lat_grid = range(lat_min, lat_max, length=resolution)

	# Perform IDW interpolation - create all grid points
	grid_lon = Float64[]
	grid_lat = Float64[]
	grid_values = Float64[]

	for lat_interp in lat_grid
		for lon_interp in lon_grid
			# Calculate distances to all data points
			distances = sqrt.((lon_clean .- lon_interp).^2 + (lat_clean .- lat_interp).^2)

			# Handle case where interpolation point coincides with data point
			min_dist = minimum(distances)
			if min_dist < 1e-10
				closest_idx = findfirst(distances .== min_dist)
				interp_value = values_clean[closest_idx]
			else
				# IDW interpolation
				weights = 1.0 ./ (distances.^power)
				interp_value = sum(weights .* values_clean) / sum(weights)
			end

			push!(grid_lon, lon_interp)
			push!(grid_lat, lat_interp)
			push!(grid_values, interp_value)
		end
	end

	# Calculate marker size for good coverage
	avg_spacing = (lon_range + lat_range) / (2 * resolution)
	marker_size = max(10, Int(round(avg_spacing * zoom * zoom * 150 * marker_scale)))

	# Create heatmap trace using large overlapping markers
	heatmap_trace = PlotlyJS.scattermapbox(
		lon=grid_lon,
		lat=grid_lat,
		mode="markers",
		marker=PlotlyJS.attr(
			size=marker_size,
			color=grid_values,
			colorscale=NMFk.colorscale(colorscale),
			opacity=opacity,
			line=PlotlyJS.attr(width=0),  # No border for smooth appearance
			colorbar=PlotlyJS.attr(
				title=title_colorbar,
				titlefont=PlotlyJS.attr(size=font_size),
				tickfont=PlotlyJS.attr(size=font_size)
			)
		),
		hovertemplate="<b>Lon:</b> %{lon}<br><b>Lat:</b> %{lat}<br><b>Value:</b> %{marker.color}<extra></extra>",
		showlegend=false,
		name="Heatmap"
	)

	traces = [heatmap_trace]

	# Add data points if requested
	if show_points
		points_trace = PlotlyJS.scattermapbox(
			lon=lon_clean,
			lat=lat_clean,
			text=["Lon: $(round(lon_clean[i], digits=4))<br>Lat: $(round(lat_clean[i], digits=4))<br>Value: $(round(values_clean[i], digits=2))" for i in eachindex(lon_clean)],
			mode="markers",
			marker=PlotlyJS.attr(
				size=point_size,
				color="white",
				line=PlotlyJS.attr(color="black", width=2)
			),
			name="Data Points",
			hoverinfo="text",
			showlegend=true
		)
		push!(traces, points_trace)
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
	distances = sqrt.((x_data .- x_interp).^2 + (y_data .- y_interp).^2)

	# Handle case where interpolation point coincides with data point
	min_dist = minimum(distances)
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

"""
	mapbox_contour_simple(df, column; kw...)

Simple version of mapbox_contour for DataFrames with guaranteed visibility.

# Arguments
- `df::DataFrames.DataFrame`: DataFrame containing longitude, latitude, and value columns
- `column::Union{Symbol, AbstractString}`: Column name containing values to interpolate
- `kw...`: Additional keyword arguments passed to mapbox_contour_simple

# Returns
- PlotlyJS plot object with simple heatmap overlay
"""
function mapbox_contour_simple(df::DataFrames.DataFrame, column::Union{Symbol, AbstractString}; kw...)
	lon, lat = get_lonlat(df)
	if isnothing(lon) || isnothing(lat)
		@error("Longitude and latitude columns are required for plotting!")
		return nothing
	end

	if !(column in names(df))
		@error("Column '$column' not found in DataFrame!")
		return nothing
	end

	return mapbox_contour_simple(lon, lat, df[!, column]; kw...)
end