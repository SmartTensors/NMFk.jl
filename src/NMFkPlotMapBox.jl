import DataFrames
import PlotlyJS
import Interpolations
import NearestNeighbors
import Statistics
import ConcaveHull
import Printf
import PlotlyKaleido

const _DEFAULT_MAPBOX_CONTOUR_RESOLUTION = 50
const _DEFAULT_MAPBOX_CONTOUR_MAX_FEATURES = 50_000
const _DEFAULT_MAPBOX_CONTOUR_HOVER_MAX_FEATURES = 5_000
const _DEFAULT_MAPBOX_CONTOUR_NEIGHBORS = 0
const _DEFAULT_MAPBOX_CONTOUR_HULL_MAX_POINTS = 10_000
const _DEFAULT_MAPBOX_CONTOUR_HULL_FORCE_CONVEX_ABOVE = 0
const _DEFAULT_MAPBOX_CONTOUR_HULL_CONCAVE_MAX_POINTS = 12_000
const _DEFAULT_MAPBOX_CONTOUR_HULL_MODE = :direct
const _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MAX_POINTS = 40_000
const _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_BUFFER_CELLS = 6
const _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MIN_POINTS = 20_000
const _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_PASSES = 3

const _DEFAULT_MAPBOX_CONTOUR_MASK_MODE = :auto
const _DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_MULTIPLIER = 2.0
const _DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_QUANTILE = 0.5
const _DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_MAX_POINTS = 5_000

function _mean_finite(v::AbstractVector{<:Real}; default::Float64=0.0)
	acc = 0.0
	n = 0
	for x in v
		xf = Float64(x)
		if isfinite(xf)
			acc += xf
			n += 1
		end
	end
	return n > 0 ? (acc / n) : default
end

function _estimate_nn_spacing(
	tree::NearestNeighbors.KDTree,
	coords::AbstractMatrix{<:Real};
	max_points::Int=_DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_MAX_POINTS,
	q::Real=_DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_QUANTILE,
)
	n = size(coords, 2)
	if n < 2
		return 0.0
	end
	qf = Float64(q)
	if !(0.0 <= qf <= 1.0)
		throw(ArgumentError("q must be in [0, 1]"))
	end
	m = max_points > 0 ? min(n, max_points) : n
	stride = n <= m ? 1 : max(1, cld(n, m))
	idxs = collect(1:stride:n)
	Q = Matrix{Float64}(undef, 2, length(idxs))
	@inbounds for (k, i) in enumerate(idxs)
		Q[1, k] = Float64(coords[1, i])
		Q[2, k] = Float64(coords[2, i])
	end
	_, d2s = NearestNeighbors.knn(tree, Q, 2, true)
	d = Float64[]
	sizehint!(d, length(d2s))
	@inbounds for k in eachindex(d2s)
		vec = d2s[k]
		if length(vec) >= 2
			d2 = vec[2]
			if isfinite(d2) && d2 > 0
				push!(d, sqrt(Float64(d2)))
			end
		end
	end
	if isempty(d)
		return 0.0
	end
	return Float64(Statistics.quantile(d, qf))
end

function _compute_bounds_and_bins(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real}; max_points::Int, grid_bins::Int=0)
	n = length(lon)
	if n == 0
		return (Inf, -Inf, Inf, -Inf, 0)
	end
	bins = grid_bins > 0 ? grid_bins : floor(Int, sqrt(max_points))
	bins = max(2, bins)
	lon_min = Inf
	lon_max = -Inf
	lat_min = Inf
	lat_max = -Inf
	count = 0
	for i in eachindex(lon)
		x = Float64(lon[i])
		y = Float64(lat[i])
		if !isfinite(x) || !isfinite(y)
			continue
		end
		count += 1
		if x < lon_min
			lon_min = x
		end
		if x > lon_max
			lon_max = x
		end
		if y < lat_min
			lat_min = y
		end
		if y > lat_max
			lat_max = y
		end
	end
	if count == 0
		return (Inf, -Inf, Inf, -Inf, 0)
	end
	return (lon_min, lon_max, lat_min, lat_max, bins)
end

function _sample_polygon_boundary(vertices::Vector{Tuple{Float64, Float64}}; step::Float64)
	if length(vertices) < 2
		return Matrix{Float64}(undef, 2, 0)
	end
	step = max(step, 1e-12)
	closed = vertices[1] == vertices[end]
	verts = closed ? vertices : vcat(vertices, vertices[1])
	# Conservative size estimate
	coords = Vector{Tuple{Float64, Float64}}()
	sizehint!(coords, max(32, length(verts) * 4))
	for i in 1:(length(verts) - 1)
		p1 = verts[i]
		p2 = verts[i + 1]
		dx = p2[1] - p1[1]
		dy = p2[2] - p1[2]
		len = hypot(dx, dy)
		if len <= 0
			push!(coords, p1)
			continue
		end
		nseg = max(1, ceil(Int, len / step))
		for s in 0:nseg
			t = s / nseg
			push!(coords, (p1[1] + t * dx, p1[2] + t * dy))
		end
	end
	m = length(coords)
	M = Matrix{Float64}(undef, 2, m)
	@inbounds for j in 1:m
		M[1, j] = coords[j][1]
		M[2, j] = coords[j][2]
	end
	return M
end

function _select_points_near_boundary(
	lon::AbstractVector{<:Real},
	lat::AbstractVector{<:Real},
	boundary_tree::NearestNeighbors.KDTree;
	max_points::Int,
	min_points::Int,
	near_dist::Float64,
)
	n = length(lon)
	if n == 0
		return Int[]
	end
	near2 = near_dist * near_dist
	d2 = Vector{Float64}(undef, n)
	chunk = 50_000
	Q = Matrix{Float64}(undef, 2, min(chunk, n))
	pos = 1
	while pos <= n
		last = min(n, pos + chunk - 1)
		len = last - pos + 1
		if size(Q, 2) != len
			Q = Matrix{Float64}(undef, 2, len)
		end
		@inbounds for j in 1:len
			Q[1, j] = Float64(lon[pos + j - 1])
			Q[2, j] = Float64(lat[pos + j - 1])
		end
		_idx, d2s = NearestNeighbors.knn(boundary_tree, Q, 1, true)
		@inbounds for j in 1:len
			d2[pos + j - 1] = d2s[j][1]
		end
		pos = last + 1
	end
	idxs = findall(x -> x <= near2, d2)
	min_points_eff = max(3, min_points)
	if length(idxs) < min_points_eff
		k = min(n, (max_points > 0 ? min(max_points, min_points_eff) : min_points_eff))
		return partialsortperm(d2, 1:k)
	end
	if max_points > 0 && length(idxs) > max_points
		return partialsortperm(d2, 1:max_points)
	end
	return idxs
end

mapbox_token = get(ENV, "MAPBOX_ACCESS_TOKEN", "")

function __init__()
	# NMFk is frequently precompiled; make sure we pick up the current runtime
	# environment token rather than whatever was present at precompile time.
	env_tok = strip(get(ENV, "MAPBOX_ACCESS_TOKEN", ""))
	if !isempty(env_tok)
		global mapbox_token = env_tok
	end
	return nothing
end

function ensure_mapbox_token!(token)
	# If an explicit token is provided, register it. Otherwise, pick up the
	# current ENV token (useful when ENV is set after module load).
	if token isa AbstractString
		tok = strip(token)
		if isempty(tok)
			env_tok = strip(get(ENV, "MAPBOX_ACCESS_TOKEN", ""))
			if !isempty(env_tok)
				global mapbox_token = env_tok
			end
			return
		end
		current = get(ENV, "MAPBOX_ACCESS_TOKEN", nothing)
		if current !== tok
			ENV["MAPBOX_ACCESS_TOKEN"] = tok
		end
		global mapbox_token = tok
	end
end

function _token_free_style(style::AbstractString)
	# Plotly supports token-free Mapbox styles like open-street-map/carto/stamen.
	return "open-street-map"
end

regex_lon = r"^[Xx]$|^[Ll]on$|^LONGITUDE$|^LON$|^[Ll]ongitude$" # regex for longitude
regex_lat = r"^[Yy]$|^[Ll]at$|^LATITUDE$|^LAT$|^[Ll]atitude$" # regex for latitude

function vector_float64(v::AbstractVector)
	out = Vector{Float64}(undef, length(v))
	@inbounds for i in eachindex(v)
		x = v[i]
		if x isa Missing
			out[i] = NaN
		elseif x isa Real
			out[i] = Float64(x)
		else
			y = tryparse(Float64, string(x))
			out[i] = isnothing(y) ? NaN : y
		end
	end
	return out
end

function vector_string(v::AbstractVector)
	out = Vector{String}(undef, length(v))
	@inbounds for i in eachindex(v)
		x = v[i]
		out[i] = x isa Missing ? "missing" : string(x)
	end
	return out
end

function safe_savefig(args...; kwargs...)
    try
        PlotlyJS.savefig(args...; kwargs...)
    catch err
        if err isa InterruptException
            @info "Save interrupted – restarting Kaleido"
            PlotlyKaleido.restart()
            rethrow()
        else
			msg = sprint(showerror, err)
			if occursin("Missing Mapbox access token", msg) || occursin("mapboxAccessToken", msg)
				@error "Plotly export failed due to missing Mapbox token. Either set ENV[\"MAPBOX_ACCESS_TOKEN\"], pass mapbox_token=... to NMFk plotting calls, or use a token-free basemap style like style=\"open-street-map\"."
			end
            rethrow()
        end
    end
end

# Mapbox for a dataframe with multiple columns
function mapbox(df::DataFrames.DataFrame; namesmap=names(df), column::Union{Symbol, AbstractString}="", filename::AbstractString="", title::AbstractString="", title_colorbar::AbstractString=title, title_length::Number=0, categorical::Bool=false, quiet::Bool=false, kw...)
	lon, lat = get_lonlat(df)
	if isnothing(lon) || isnothing(lat)
		@error("Longitude and latitude columns are required for plotting!")
		return nothing
	end
	lonf = vector_float64(lon)
	latf = vector_float64(lat)
	fileroot, fileext = splitext(filename)
	if column == ""
		local col = 1
		for a in names(df)
			if !(occursin(regex_lon, a) || occursin(regex_lat, a))
				varname = namesmap[col]
				col += 1
				println("Plotting '$(varname)' ...")
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
					p = mapbox(lonf, latf, vector_string(df[!, a]); filename=f, title_colorbar=t, title=title, quiet=true, kw...)
				else
					p = mapbox(lonf, latf, vector_float64(df[!, a]); filename=f, title_colorbar=t, title=title, quiet=true, kw...)
				end
				!quiet && display(p)
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
		if categorical
			p = mapbox(lonf, latf, vector_string(df[!, column]); filename=f, title_colorbar=plotly_title_length(column, title_length), title=title, quiet=true, kw...)
		else
			p = mapbox(lonf, latf, vector_float64(df[!, column]); filename=f, title_colorbar=plotly_title_length(column, title_length), title=title, quiet=true, kw...)
		end
		!quiet && display(p)
	end
	return nothing
end

# Mapbox for a matrix with multiple columns
function mapbox(lon::AbstractVector{T1}, lat::AbstractVector{T1}, M::AbstractMatrix{T2}, names::AbstractVector=["Column $i" for i in axes(M, 2)]; filename::AbstractString="", title::AbstractString="", title_colorbar::AbstractString=title, title_length::Number=0, quiet::Bool=false, kw...) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	fileroot, fileext = splitext(filename)
	for i in eachindex(names)
		println("Plotting '$(names[i])' ...")
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
		p = mapbox(lon, lat, M[:, i]; filename=f, title_colorbar=t, title=title, quiet=true, kw...)
		!quiet && display(p)
	end
	return nothing
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
	text::AbstractVector=[],
	lon_center::AbstractFloat=minimumnan(lon) + (maximumnan(lon) - minimumnan(lon)) / 2,
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
	lat_center::AbstractFloat=minimumnan(lat) + (maximumnan(lat) - minimumnan(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	zoom_fig::Number=zoom,
	dot_size::Number=compute_dot_size(lon, lat, zoom),
	dot_size_fig::Number=dot_size * 2,
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	filename::AbstractString="",
	figuredir::AbstractString=".",
	format::AbstractString=splitext(filename)[end][2:end],
	width::Int=14,
	height::Int=7,
	dpi::Int=200,
	width_pixel::Int=dpi * width,
	height_pixel::Int=dpi * height,
	display_width_pixel::Union{Int,Nothing}=nothing,
	display_height_pixel::Union{Int,Nothing}=nothing,
	scale::Real=1,
	legend::Bool=true,
	colorbar::Bool=legend,
	traces::AbstractVector=[],
	traces_setup=(; mode="lines", line_width=8, line_color="purple", attributionControl=false),
	colorscale::Symbol=:turbo,
	colorbar_bgcolor::AbstractString="#5a5a5a",
	colorbar_bgcolor_fig::AbstractString=colorbar_bgcolor,
	colorbar_font_color::AbstractString="white",
	colorbar_font_color_fig::AbstractString=colorbar_font_color,
	colorbar_font_family::AbstractString="Arial",
	colorbar_font_family_fig::AbstractString=colorbar_font_family,
	colorbar_font_size::Number=font_size,
	colorbar_font_size_fig::Number=font_size_fig,
	colorbar_font_bold::Bool=true,
	colorbar_font_bold_fig::Bool=colorbar_font_bold,
	paper_bgcolor::AbstractString=colorbar_bgcolor,
	paper_bgcolor_fig::AbstractString=paper_bgcolor,
	quiet::Bool=false,
	show_count::Bool=true, # dummy
	show_locations::Bool=false, # dummy
	preset::Symbol=:none, # dummy
) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	if length(text) > 0
		@assert length(lon) == length(text)
	else
		showlabels = false
	end
	ensure_mapbox_token!(mapbox_token)
	if title == title_colorbar
		title = ""
	end
	traces = check_traces(traces, traces_setup)
	sort_color = sortpermnan(color)
	if filename != ""
		show_colorbar_fig = style_mapbox_traces!(traces, legend; line_color=line_color, line_width=line_width_fig, marker_color=marker_color, marker_size=marker_size_fig)
		if colorbar && show_colorbar_fig
			colorbar_attr = mapbox_colorbar_attr(title_colorbar, title_length; font_size=colorbar_font_size_fig, font_color=colorbar_font_color_fig, font_family=colorbar_font_family_fig, bgcolor=colorbar_bgcolor_fig, bold=colorbar_font_bold_fig)
		else
			colorbar_attr = PlotlyJS.attr()
		end
		plot = build_scatter_trace(lon, lat, text, color, sort_color; dot_size=dot_size_fig, showlabels=showlabels, label_position=label_position, label_font_size=label_font_size_fig, label_font_color=label_font_color_fig, colorbar_attr=colorbar_attr, zmin=zmin, zmax=zmax, colorscale=colorscale)
		layout = plotly_layout(lon_center, lat_center, zoom_fig; width=width_pixel, height=height_pixel, title=title, font_size=font_size_fig, style=style, paper_bgcolor=paper_bgcolor_fig, mapbox_token=mapbox_token)
		p = PlotlyJS.plot([plot, traces...], layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		safe_savefig(p, fn; format=format, width=width_pixel, height=height_pixel, scale=scale)
	end
	show_colorbar = style_mapbox_traces!(traces, legend; line_color=line_color, line_width=line_width, marker_color=marker_color, marker_size=marker_size)
	if colorbar && show_colorbar
		colorbar_attr = mapbox_colorbar_attr(title_colorbar, title_length; font_size=colorbar_font_size, font_color=colorbar_font_color, font_family=colorbar_font_family, bgcolor=colorbar_bgcolor, bold=colorbar_font_bold)
	else
		colorbar_attr = PlotlyJS.attr()
	end
	plot = build_scatter_trace(lon, lat, text, color, sort_color; dot_size=dot_size, showlabels=showlabels, label_position=label_position, label_font_size=label_font_size, label_font_color=label_font_color, colorbar_attr=colorbar_attr, zmin=zmin, zmax=zmax, colorscale=colorscale)
	layout = plotly_layout(lon_center, lat_center, zoom; paper_bgcolor=paper_bgcolor, title=title, font_size=font_size, style=style, mapbox_token=mapbox_token, width=display_width_pixel, height=display_height_pixel)
	responsive = isnothing(display_width_pixel) && isnothing(display_height_pixel)
	p = PlotlyJS.plot([plot, traces...], layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=responsive))
	!quiet && display(p)
	return p
end

function mapbox(
	lon::AbstractVector{T1},
	lat::AbstractVector{T1},
	color::AbstractVector{T2};
	title::AbstractString="",
	title_colorbar::AbstractString="",
	title_length::Number=0,
	text::AbstractVector=[],
	lon_center::AbstractFloat=minimumnan(lon) + (maximumnan(lon) - minimumnan(lon)) / 2,
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
	lat_center::AbstractFloat=minimumnan(lat) + (maximumnan(lat) - minimumnan(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	zoom_fig::Number=zoom,
	dot_size::Number=compute_dot_size(lon, lat, zoom),
	dot_size_fig::Number=dot_size * 2,
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	filename::AbstractString="",
	figuredir::AbstractString=".",
	format::AbstractString=splitext(filename)[end][2:end],
	width::Int=14,
	height::Int=7,
	dpi::Int=200,
	width_pixel::Int=dpi * width,
	height_pixel::Int=dpi * height,
	display_width_pixel::Union{Int,Nothing}=nothing,
	display_height_pixel::Union{Int,Nothing}=nothing,
	scale::Real=1,
	legend::Bool=true,
	colorbar::Bool=legend,
	traces::AbstractVector=[],
	traces_setup=(; mode="lines", line_width=8, line_color="purple", attributionControl=false),
	showlegend::Bool=false, # dummy
	show_locations::Bool=false, # dummy
	colorscale::Symbol=:turbo, # dummy
	preset::Symbol=:none, # dummy
	quiet::Bool=false,
	paper_bgcolor::AbstractString="white",
	show_count::Bool=true
) where {T1 <: AbstractFloat, T2 <: Union{Number, Symbol, AbstractString, AbstractChar}}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	if length(text) > 0
		@assert length(lon) == length(text)
	else
		showlabels = false
	end
	ensure_mapbox_token!(mapbox_token)
	if width_pixel < 500 || height_pixel < 500
		@error "Width and height in pixels should be at least 500 for proper rendering" width_pixel=width_pixel height_pixel=height_pixel
		throw(ArgumentError("Insufficient width/height in pixels"))
	end
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
			name = show_count ? "$(string(i)) [$(sum(iz))]" : "$(string(i))"
			t = PlotlyJS.scattermapbox(;
				lon=lon[iz],
				lat=lat[iz],
				text=showlabels ? text[iz] : nothing,
				hoverinfo=showlabels ? "text" : "",
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
		layout = plotly_layout(lon_center, lat_center, zoom_fig; paper_bgcolor=paper_bgcolor, font_size=font_size_fig, font_color=font_color_fig, title=title, style=style, mapbox_token=mapbox_token)
		p = PlotlyJS.plot(traces_, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		@info("Saving map as '$(fn)' with format '$(format)' width $(width_pixel) height $(height_pixel) and scale $(scale) ...")
		safe_savefig(p, fn; format=format, width=width_pixel, height=height_pixel, scale=scale)
	end
	traces_ = Vector{PlotlyJS.GenericTrace{Dict{Symbol, Any}}}(undef, 0)
	for (j, i) in enumerate(unique(sort(color)))
		iz = color .== i
		jj = j % length(NMFk.colors)
		k = jj == 0 ? length(NMFk.colors) : jj
		marker = PlotlyJS.attr(; size=dot_size, color=NMFk.colors[k], colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(repeat("&nbsp;", title_length) * " colorbar " * title, title_length), titlefont=PlotlyJS.attr(; size=font_size, color=paper_bgcolor), tickfont=PlotlyJS.attr(; size=font_size, color=paper_bgcolor)))
		name = show_count ? "$(string(i)) [$(sum(iz))]" : "$(string(i))"
		t = PlotlyJS.scattermapbox(;
			lon=lon[iz],
			lat=lat[iz],
			text=showlabels ? text[iz] : nothing,
			hoverinfo=showlabels ? "text" : "",
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
	layout = plotly_layout(lon_center, lat_center, zoom; paper_bgcolor=paper_bgcolor, title=title, font_size=font_size, font_color=font_color, style=style, mapbox_token=mapbox_token, width=display_width_pixel, height=display_height_pixel)
	responsive = isnothing(display_width_pixel) && isnothing(display_height_pixel)
	p = PlotlyJS.plot(traces_, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=responsive))
	!quiet && display(p)
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
function plotly_layout(
	lon_center::Number=-105.9378,
	lat_center::Number=35.6870,
	zoom::Number=4;
	paper_bgcolor::AbstractString="white",
	width::Union{Int,Nothing}=nothing,
	height::Union{Int,Nothing}=nothing,
	title::AbstractString="",
	font_size::Number=14,
	font_color="black",
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	margin=PlotlyJS.attr(; r=0, t=0, b=0, l=0),
	autosize::Union{Bool,Nothing}=nothing,
)
	style_str = String(style)
	token_str = mapbox_token isa AbstractString ? strip(mapbox_token) : ""
	if isempty(token_str)
		# If caller didn't pass a token (or passed an empty token), fall back to ENV.
		token_str = strip(get(ENV, "MAPBOX_ACCESS_TOKEN", ""))
	end
	if isempty(token_str) && startswith(style_str, "mapbox://")
		# Mapbox-hosted styles require a token; fall back to a token-free basemap.
		@warn "Mapbox style requested but MAPBOX_ACCESS_TOKEN is missing; falling back to a token-free basemap style" requested_style=style_str fallback_style=_token_free_style(style_str)
		style_str = _token_free_style(style_str)
	end
	mapbox_attr = PlotlyJS.attr(
		; style=style_str,
		center=PlotlyJS.attr(; lon=lon_center, lat=lat_center),
		zoom=zoom,
	)
	if !isempty(token_str)
		mapbox_attr[:accesstoken] = token_str
	end
	# If width/height are not provided, let the frontend (e.g., VS Code plots panel)
	# size the plot; this avoids huge default canvases that appear "oversized".
	local_autosize = isnothing(autosize) ? (isnothing(width) && isnothing(height)) : autosize
	base = (
		margin=margin,
		autosize=local_autosize,
		legend=PlotlyJS.attr(; title_text=title, title_font_size=font_size, itemsizing="constant", font=PlotlyJS.attr(; size=font_size, color=font_color), bgcolor=paper_bgcolor),
		paper_bgcolor=paper_bgcolor,
		mapbox=mapbox_attr,
	)
	if isnothing(width) && isnothing(height)
		return PlotlyJS.Layout(; base...)
	end
	extra = (;)
	if !isnothing(width)
		extra = merge(extra, (; width=width))
	end
	if !isnothing(height)
		extra = merge(extra, (; height=height))
	end
	# If explicit dimensions are provided, autosize should normally be false.
	if isnothing(autosize)
		extra = merge(extra, (; autosize=false))
	end
	return PlotlyJS.Layout(; base..., extra...)
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

const _DEFAULT_COLORBAR_THICKNESS = 30
const _DEFAULT_COLORBAR_LEN = 0.5

function mapbox_colorbar_attr(
	title::AbstractString,
	title_length::Number;
	font_size::Number,
	font_color::AbstractString,
	font_family::AbstractString,
	bgcolor::AbstractString,
	thickness::Number=_DEFAULT_COLORBAR_THICKNESS,
	len::Real=_DEFAULT_COLORBAR_LEN,
	bold::Bool=true,
	kwargs...
)
	processed_title = plotly_title_length(title, title_length)
	if !isempty(strip(processed_title))
		processed_title *= "<br>&nbsp;<br>"
	end
	bold_title = bold ? "<b>" * processed_title * "</b>" : processed_title
	base_family = isempty(strip(font_family)) ? "Arial" : font_family
	effective_family = if bold
		occursin(r"(?i)bold", base_family) ? base_family : string(base_family, " Bold, ", base_family)
	else
		base_family
	end
	effective_family = string(effective_family, ", sans-serif")
	title_font = PlotlyJS.attr(; size=font_size, color=font_color, family=effective_family)
	tick_font = PlotlyJS.attr(; size=font_size, color=font_color, family=effective_family)
	return PlotlyJS.attr(
		; thicknessmode="pixels",
		thickness=thickness,
		len=len,
		bgcolor=bgcolor,
		title=bold_title,
		titlefont=title_font,
		tickfont=tick_font,
		kwargs...
	)
end

function style_mapbox_traces!(traces::AbstractVector, legend::Bool; line_color::AbstractString="", line_width::Number=0, marker_color::AbstractString="", marker_size::Number=0)
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
	return show_colorbar
end

function build_scatter_trace(
	lon::AbstractVector,
	lat::AbstractVector,
	text::AbstractVector,
	color::AbstractVector,
	sort_idx::AbstractVector{<:Integer};
	dot_size::Number,
	showlabels::Bool,
	label_position::AbstractString,
	label_font_size::Number,
	label_font_color::AbstractString,
	colorbar_attr,
	zmin::Number,
	zmax::Number,
	colorscale::Symbol
)
	marker = PlotlyJS.attr(; size=dot_size, color=color[sort_idx], colorscale=NMFk.colorscale(colorscale), cmin=zmin, cmax=zmax, colorbar=colorbar_attr)
	return PlotlyJS.scattermapbox(; lon=lon[sort_idx], lat=lat[sort_idx], text=showlabels ? text[sort_idx] : [], mode=showlabels ? "markers+text" : "markers", hoverinfo="text", marker=marker, textposition=label_position, textfont=PlotlyJS.attr(; size=label_font_size, color=label_font_color), attributionControl=false, showlegend=false)
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

function _colorbar_tick_values(zmin::Float64, zmax::Float64; count::Int=5)
	if !isfinite(zmin) || !isfinite(zmax)
		return [zmin, zmax]
	end
	if isapprox(zmin, zmax; atol=1e-12, rtol=1e-12)
		return [zmin]
	end
	count = max(2, count)
	return collect(range(zmin, zmax; length=count))
end


_strip_trailing_zeros(str::AbstractString) = begin
	pos = Base.findfirst(==('.'), str)
	if isnothing(pos)
		return str
	end
	trimmed = replace(str, r"0+$" => "")
	trimmed = replace(trimmed, r"\.$" => "")
	return isempty(trimmed) ? "0" : trimmed
end

function _format_decimal_label(val::Float64)
	!isfinite(val) && return string(val)
	str = Printf.@sprintf("%.6f", val)
	return _strip_trailing_zeros(str)
end

function _format_scientific_label(val::Float64)
	!isfinite(val) && return string(val)
	val == 0 && return "0"
	str = lowercase(Printf.@sprintf("%.4e", val))
	parts = split(str, 'e')
	mant = _strip_trailing_zeros(parts[1])
	exp = parse(Int, parts[2])
	return string(mant, "e", exp)
end

function _colorbar_tick_labels(values::Vector{Float64})
	finite_vals = filter(isfinite, values)
	formatter = if isempty(finite_vals)
		val -> string(val)
	elseif any(v -> abs(v) >= 1e4 || (abs(v) > 0 && abs(v) < 1e-2), finite_vals)
		_format_scientific_label
	else
		_format_decimal_label
	end
	return [formatter(val) for val in values]
end

function _grid_thin_indices(
	lon::AbstractVector{<:Real},
	lat::AbstractVector{<:Real};
	max_points::Int,
	grid_bins::Int=0,
)
	n = length(lon)
	if max_points <= 0 || n <= max_points
		return collect(eachindex(lon))
	end
	# Pick (approximately) one representative point per grid cell.
	# To preserve the boundary better, keep the point farthest from the
	# global centroid within each occupied cell.
	bins = grid_bins > 0 ? grid_bins : floor(Int, sqrt(max_points))
	bins = max(2, bins)

	# Compute bounding box + centroid without allocating large temporaries.
	lon_min = Inf
	lon_max = -Inf
	lat_min = Inf
	lat_max = -Inf
	sumx = 0.0
	sumy = 0.0
	count = 0
	iminlon = 0
	imaxlon = 0
	iminlat = 0
	imaxlat = 0
	for i in eachindex(lon)
		x = Float64(lon[i])
		y = Float64(lat[i])
		if !isfinite(x) || !isfinite(y)
			continue
		end
		count += 1
		sumx += x
		sumy += y
		if x < lon_min
			lon_min = x
			iminlon = Int(i)
		end
		if x > lon_max
			lon_max = x
			imaxlon = Int(i)
		end
		if y < lat_min
			lat_min = y
			iminlat = Int(i)
		end
		if y > lat_max
			lat_max = y
			imaxlat = Int(i)
		end
	end
	if count == 0
		return Int[]
	end
	cx = sumx / count
	cy = sumy / count
	span_lon = max(lon_max - lon_min, 1e-12)
	span_lat = max(lat_max - lat_min, 1e-12)
	cell_lon = span_lon / bins
	cell_lat = span_lat / bins
	# cell key -> (index, score)
	seen = Dict{UInt64, Tuple{Int, Float64}}()
	for i in eachindex(lon)
		x = Float64(lon[i])
		y = Float64(lat[i])
		if !isfinite(x) || !isfinite(y)
			continue
		end
		ix = clamp(floor(Int, (x - lon_min) / cell_lon) + 1, 1, bins)
		iy = clamp(floor(Int, (y - lat_min) / cell_lat) + 1, 1, bins)
		key = (UInt64(ix) << 32) | UInt64(iy)
		score = (x - cx) * (x - cx) + (y - cy) * (y - cy)
		if haskey(seen, key)
			idx_prev, score_prev = seen[key]
			if score > score_prev
				seen[key] = (Int(i), score)
			end
		else
			seen[key] = (Int(i), score)
		end
	end
	idxs = Int[]
	sizehint!(idxs, min(max_points, length(seen) + 8))
	for (_k, (idx, _score)) in seen
		push!(idxs, idx)
	end
	# Ensure we keep the extreme points (helps stabilize hull for sparse bins)
	for ii in (iminlon, imaxlon, iminlat, imaxlat)
		ii > 0 && push!(idxs, ii)
	end
	return unique(idxs)
end

function _prepare_hull_points(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real}; max_points::Int=0, grid_bins::Int=0)
	idxs = _grid_thin_indices(lon, lat; max_points=max_points, grid_bins=grid_bins)
	coords = Vector{Tuple{Float64, Float64}}()
	sizehint!(coords, length(idxs))
	for i in idxs
		x = Float64(lon[i])
		y = Float64(lat[i])
		if isfinite(x) && isfinite(y)
			push!(coords, (x, y))
		end
	end
	return unique(coords)
end

function compute_convex_hull_vertices(coords::Vector{Tuple{Float64, Float64}})
	if length(coords) < 3
		return nothing
	end
	sorted = sort(coords; by=p -> (p[1], p[2]))
	cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
	lower = Tuple{Float64, Float64}[]
	for p in sorted
		while length(lower) >= 2 && cross(lower[end - 1], lower[end], p) <= 0
			pop!(lower)
		end
		push!(lower, p)
	end
	upper = Tuple{Float64, Float64}[]
	for p in reverse(sorted)
		while length(upper) >= 2 && cross(upper[end - 1], upper[end], p) <= 0
			pop!(upper)
		end
		push!(upper, p)
	end
	hull = vcat(lower[1:end - 1], upper[1:end - 1])
	if isempty(hull)
		return nothing
	end
	if hull[1] != hull[end]
		push!(hull, hull[1])
	end
	return hull
end


_orientation(a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64}, c::Tuple{Float64, Float64}) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

function _point_between(a::Tuple{Float64, Float64}, b::Tuple{Float64, Float64}, c::Tuple{Float64, Float64}; eps::Float64=1e-12)
	min_x, max_x = min(a[1], b[1]) - eps, max(a[1], b[1]) + eps
	min_y, max_y = min(a[2], b[2]) - eps, max(a[2], b[2]) + eps
	return min_x <= c[1] <= max_x && min_y <= c[2] <= max_y
end

function _segments_intersect(p1, p2, p3, p4; eps::Float64=1e-12)
	o1 = _orientation(p1, p2, p3)
	o2 = _orientation(p1, p2, p4)
	o3 = _orientation(p3, p4, p1)
	o4 = _orientation(p3, p4, p2)
	if (o1 * o2 < -eps) && (o3 * o4 < -eps)
		return true
	end
	if abs(o1) <= eps && _point_between(p1, p2, p3; eps=eps)
		return true
	end
	if abs(o2) <= eps && _point_between(p1, p2, p4; eps=eps)
		return true
	end
	if abs(o3) <= eps && _point_between(p3, p4, p1; eps=eps)
		return true
	end
	if abs(o4) <= eps && _point_between(p3, p4, p2; eps=eps)
		return true
	end
	return false
end

function _polygon_self_intersects(vertices::Vector{Tuple{Float64, Float64}}; eps::Float64=1e-12)
	n = length(vertices)
	n < 4 && return false
	n_edges = vertices[1] == vertices[end] ? n - 1 : n
	if n_edges < 4
		return false
	end
	for i in 1:n_edges
		p1 = vertices[i]
		p2 = vertices[i == n_edges ? 1 : i + 1]
		for j in i+1:n_edges
			if abs(i - j) <= 1
				continue
			end
			if i == 1 && j == n_edges
				continue
			end
			q1 = vertices[j]
			q2 = vertices[j == n_edges ? 1 : j + 1]
			if _segments_intersect(p1, p2, q1, q2; eps=eps)
				return true
			end
		end
	end
	return false
end

function compute_concave_hull_vertices(
	lon::AbstractVector{<:Real},
	lat::AbstractVector{<:Real};
	convex::Bool=false,
	mode::Symbol=_DEFAULT_MAPBOX_CONTOUR_HULL_MODE,
	max_points::Int=0,
	grid_bins::Int=0,
	concave_max_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_CONCAVE_MAX_POINTS,
	stepwise_max_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MAX_POINTS,
	stepwise_min_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MIN_POINTS,
	stepwise_buffer_cells::Real=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_BUFFER_CELLS,
	stepwise_passes::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_PASSES,
)
	function _direct_concave_from_coords(coords_local::Vector{Tuple{Float64, Float64}})
		if length(coords_local) < 3
			return nothing
		end
		# ConcaveHull can become unstable/slow for very large input sizes; for
		# robustness, re-thin specifically for the concave hull stage.
		if concave_max_points > 0 && length(coords_local) > concave_max_points
			lon2 = Vector{Float64}(undef, length(coords_local))
			lat2 = Vector{Float64}(undef, length(coords_local))
			@inbounds for i in eachindex(coords_local)
				lon2[i] = coords_local[i][1]
				lat2[i] = coords_local[i][2]
			end
			idxs2 = _grid_thin_indices(lon2, lat2; max_points=concave_max_points, grid_bins=grid_bins)
			coords_local = coords_local[idxs2]
			@warn "Concave hull input thinned for stability" n_in=length(lon2) n_used=length(coords_local) concave_max_points=concave_max_points
		end
		pts = [[p[1], p[2]] for p in coords_local]
		try
			hull = ConcaveHull.concave_hull(pts)
			verts = [(Float64(p[1]), Float64(p[2])) for p in hull.vertices]
			if isempty(verts)
				@warn "Concave hull returned empty; falling back to convex hull"
				return compute_convex_hull_vertices(coords_local)
			end
			if verts[1] != verts[end]
				push!(verts, verts[1])
			end
			if _polygon_self_intersects(verts)
				@warn "Concave hull self-intersection detected; falling back to convex hull"
				return compute_convex_hull_vertices(coords_local)
			end
			return verts
		catch e
			@warn "Concave hull computation failed; falling back to convex hull" error=e
			return compute_convex_hull_vertices(coords_local)
		end
	end

	coords = _prepare_hull_points(lon, lat; max_points=max_points, grid_bins=grid_bins)
	if convex
		@info "Convex hull requested; skipping concave hull computation"
		return compute_convex_hull_vertices(coords)
	end
	if mode != :direct && mode != :stepwise
		throw(ArgumentError("Unknown hull mode=$(mode). Supported: :direct, :stepwise"))
	end
	if mode == :stepwise
		stepwise_passes < 1 && (stepwise_passes = 1)
		# Base hull for the first pass: convex hull on a thinned representation.
		base_vertices = compute_convex_hull_vertices(coords)
		if base_vertices === nothing || length(base_vertices) < 3
			@warn "Stepwise hull: convex hull unavailable; falling back to direct concave hull"
			return _direct_concave_from_coords(coords)
		end
		# Select boundary-near points (most interior points are irrelevant) and refine
		# the boundary estimate for additional passes.
		for pass in 1:stepwise_passes
			lon_min, lon_max, lat_min, lat_max, bins = _compute_bounds_and_bins(lon, lat; max_points=max_points, grid_bins=grid_bins)
			if !(bins > 0 && isfinite(lon_min) && isfinite(lon_max) && isfinite(lat_min) && isfinite(lat_max))
				@warn "Stepwise hull: invalid bounds; falling back to direct concave hull"
				return _direct_concave_from_coords(coords)
			end
			span_lon = max(lon_max - lon_min, 1e-12)
			span_lat = max(lat_max - lat_min, 1e-12)
			cell_lon = span_lon / bins
			cell_lat = span_lat / bins
			cell_diag = hypot(cell_lon, cell_lat)
			# Avoid too-tiny thresholds on wide domains: ensure a minimum fraction of extent.
			min_near = 0.005 * max(span_lon, span_lat)
			near_dist = max(Float64(stepwise_buffer_cells) * max(cell_diag, 1e-12), min_near)
			boundary_step = max(near_dist / 2, cell_diag / 2)
			boundary_pts = _sample_polygon_boundary(base_vertices; step=boundary_step)
			if size(boundary_pts, 2) <= 1
				@warn "Stepwise hull: boundary sampling failed; falling back to direct concave hull"
				return _direct_concave_from_coords(coords)
			end
			tree = NearestNeighbors.KDTree(boundary_pts)
			idxs = _select_points_near_boundary(lon, lat, tree; max_points=stepwise_max_points, min_points=stepwise_min_points, near_dist=near_dist)
			if length(idxs) < 3
				@warn "Stepwise hull: subset too small; falling back to direct concave hull" n_subset=length(idxs)
				return _direct_concave_from_coords(coords)
			end
			coords_subset = _prepare_hull_points(lon[idxs], lat[idxs]; max_points=max_points, grid_bins=grid_bins)
			@info "Stepwise hull: pass subset selected" pass=pass passes=stepwise_passes n_all=length(lon) n_subset=length(idxs) n_thin=length(coords_subset) near_dist=near_dist
			if pass < stepwise_passes
				# Update base hull for the next pass using a concave hull on the subset.
				updated = _direct_concave_from_coords(coords_subset)
				base_vertices = updated === nothing ? compute_convex_hull_vertices(coords_subset) : updated
				if base_vertices === nothing || length(base_vertices) < 3
					@warn "Stepwise hull: refinement hull failed; stopping early" pass=pass
					return _direct_concave_from_coords(coords_subset)
				end
			else
				return _direct_concave_from_coords(coords_subset)
			end
		end
		return _direct_concave_from_coords(coords)
	end

	return _direct_concave_from_coords(coords)
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

function _build_geojson_tiles(
	lon_grid::AbstractVector{<:Real},
	lat_grid::AbstractVector{<:Real},
	values::AbstractMatrix{<:Real};
	mask_polygon::Union{Nothing, Vector{Tuple{Float64, Float64}}}=nothing,
	mask_mode::Symbol=:hull,
	mask_tree::Union{Nothing, NearestNeighbors.KDTree}=nothing,
	mask_max_distance::Real=Inf,
	mask_lon_scale::Real=1.0,
	progress::Bool=false,
	progress_every::Int=0,
)
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
	sizehint!(features, max(0, max_i * max_j))
	sizehint!(feature_ids, max(0, max_i * max_j))
	sizehint!(tile_values, max(0, max_i * max_j))
	stride = progress_every > 0 ? progress_every : max(1, cld(max_i, 10))
	if progress
		@info "[mapbox_contour] Building GeoJSON tiles" rows=max_i cols=max_j expected=max_i*max_j
	end
	q = Vector{Float64}(undef, 2)
	use_distance_mask = (mask_mode == :distance) && (mask_tree !== nothing) && isfinite(Float64(mask_max_distance))
	use_polygon_mask = (mask_mode in (:hull, :polygon)) && (mask_polygon !== nothing)
	max_d2 = use_distance_mask ? (Float64(mask_max_distance) * Float64(mask_max_distance)) : Inf
	for i = 1:max_i
		if progress && (i == 1 || i % stride == 0 || i == max_i)
			@info "[mapbox_contour] GeoJSON tiling progress" row=i total_rows=max_i pct=round(100 * i / max_i; digits=1) kept=length(features)
		end
		for j = 1:max_j
			val = Float64(values[i, j])
			if !isfinite(val)
				continue
			end
			center_lon = (lon_edges[j] + lon_edges[j + 1]) / 2
			center_lat = (lat_edges[i] + lat_edges[i + 1]) / 2
			if use_polygon_mask
				center = (center_lon, center_lat)
				if !_point_in_polygon(center, mask_polygon)
					continue
				end
			elseif use_distance_mask
				q[1] = Float64(center_lon) * Float64(mask_lon_scale)
				q[2] = Float64(center_lat)
				_, d2s = NearestNeighbors.knn(mask_tree, q, 1, true)
				d2 = d2s[1]
				if !(isfinite(d2) && d2 <= max_d2)
					continue
				end
			elseif mask_mode == :none
				# keep all finite tiles
			elseif mask_mode != :hull && mask_mode != :polygon && mask_mode != :distance
				throw(ArgumentError("mask_mode must be :hull, :distance, or :none"))
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

function _nicenum(x::Float64, do_round::Bool)
	if !isfinite(x) || x == 0.0
		return 0.0
	end
	exp = floor(log10(abs(x)))
	f = abs(x) / 10.0^exp
	nf = if do_round
		f < 1.5 ? 1.0 : (f < 3.0 ? 2.0 : (f < 7.0 ? 5.0 : 10.0))
	else
		f <= 1.0 ? 1.0 : (f <= 2.0 ? 2.0 : (f <= 5.0 ? 5.0 : 10.0))
	end
	return copysign(nf * 10.0^exp, x)
end

function _nice_color_bounds(
	zmin::Float64,
	zmax::Float64;
	tick_count::Int=5,
	keep_min::Bool=false,
	keep_max::Bool=false
)
	if !isfinite(zmin) || !isfinite(zmax) || zmax <= zmin
		return zmin, zmax
	end
	tick_count = max(2, tick_count)
	range = zmax - zmin
	nice_range = _nicenum(range, false)
	step = _nicenum(nice_range / (tick_count - 1), true)
	if step <= 0 || !isfinite(step)
		return zmin, zmax
	end
	nice_min = floor(zmin / step) * step
	nice_max = ceil(zmax / step) * step
	if nice_max <= nice_min
		nice_max = nice_min + step
	end
	out_min = keep_min ? zmin : nice_min
	out_max = keep_max ? zmax : nice_max
	if out_max <= out_min
		out_max = out_min + step
	end
	return out_min, out_max
end

function _resolve_color_bounds(
	zmin_input::Union{Number, Nothing},
	zmax_input::Union{Number, Nothing},
	values::AbstractVector{<:Real};
	nice::Bool=true,
	tick_count::Int=5
)
	keep_min = zmin_input !== nothing
	keep_max = zmax_input !== nothing
	zmin_target = zmin_input === nothing ? NaN : float(zmin_input)
	zmax_target = zmax_input === nothing ? NaN : float(zmax_input)
	if !isfinite(zmin_target)
		candidate = minimumnan(values)
		if isfinite(candidate)
			zmin_target = float(candidate)
		elseif isfinite(zmax_target)
			zmin_target = float(zmax_target) - max(1e-6, abs(float(zmax_target)) * eps(float(zmax_target)))
		else
			zmin_target = 0.0
		end
		keep_min = false
	end
	if !isfinite(zmax_target)
		candidate = maximumnan(values)
		if isfinite(candidate)
			zmax_target = float(candidate)
		else
			zmax_target = zmin_target + 1.0
		end
		keep_max = false
	end
	if !isfinite(zmin_target)
		zmin_target = 0.0
		keep_min = false
	end
	if !isfinite(zmax_target)
		zmax_target = zmin_target + 1.0
		keep_max = false
	end
	if zmax_target <= zmin_target
		@warn "zmax must be greater than zmin; adjusting automatically" zmin=zmin_target zmax=zmax_target
		delta = max(1e-9, abs(zmin_target) * eps(zmin_target))
		zmax_target = zmin_target + (delta == 0 ? 1e-9 : delta)
		keep_max = false
	end
	if nice
		zmin_target, zmax_target = _nice_color_bounds(zmin_target, zmax_target; tick_count=tick_count, keep_min=keep_min, keep_max=keep_max)
	end
	return zmin_target, zmax_target
end

"""
mapbox_contour(lon, lat, values; resolution=50, power=2, smoothing=0.0, filename="", kw...)

Create GeoJSON-based continuous contour heatmap using IDW (Inverse Distance Weighting) interpolation.

# Arguments
- `preset::Symbol=:balanced`: Preset tuning for large datasets. Supported: `:fast`, `:balanced`, `:quality`
- `lon::AbstractVector`: Vector of longitude coordinates
- `lat::AbstractVector`: Vector of latitude coordinates
- `values::AbstractVector`: Vector of values to interpolate
- `resolution::Int=50`: Grid resolution for interpolation (higher = smoother but slower)
- `max_features::Int=50000`: Maximum number of GeoJSON tiles to emit; resolution is auto-reduced to stay within this limit
- `hover_max_features::Int=5000`: Disable hover tooltips for the interpolated layer when the number of tiles exceeds this threshold (0 disables hover always)
- `neighbors::Int=0`: Use KNN-IDW interpolation with this many neighbors (0 = auto for large datasets)
- `power::Real=2`: IDW power parameter (higher = more localized interpolation)
- `smoothing::Real=0.0`: Smoothing parameter for interpolation
- `contour_levels::Int=10`: Number of contour levels
- `filename::AbstractString=""`: Output filename for saving the plot
- `title::AbstractString=""`: Plot title
- `title_colorbar::AbstractString=title`: Colorbar title
- `title_length::Number=0`: Non-breaking space padding inserted ahead of the colorbar title to widen the right margin
- `colorscale::Symbol=:turbo`: Color scale for the heatmap
- `opacity::Real=0.7`: Opacity of the contour layer
- `colorbar_bgcolor::AbstractString="#5a5a5a"`: Background color for the colorbar
- `colorbar_font_color::AbstractString="white"`: Color for colorbar title and tick labels
- `colorbar_font_family::AbstractString="Arial"`: Base font family for colorbar text
- `colorbar_font_bold::Bool=true`: Whether to render colorbar title/tick labels in bold
- `colorbar_font_size::Number=font_size`: Font size for the colorbar title and ticks
- `paper_bgcolor::AbstractString=colorbar_bgcolor`: Canvas background color for the plot
- `show_points::Bool=false`: Whether to show original data points
- `concave_hull::Bool=true`: If true, derive extent/masking from a ConcaveHull envelope
- `hull_padding::Real=0.02`: Fractional padding applied to the concave hull shape itself
- `extra_margin::Real=0.0`: Absolute degree margin added radially outside the hull
- `hull_max_points::Int=10000`: If the point set is larger, thin points before hull computation (grid-based) for speed
- `hull_grid_bins::Int=0`: Optional fixed number of grid bins used for hull thinning (0 = auto from `hull_max_points`)
- `hull_force_convex_above::Int=200000`: If the point set exceeds this size, skip concave hull and use convex hull
- `hull_concave_max_points::Int=12000`: Safety cap for ConcaveHull input size; concave hull is computed on a thinned subset when needed
- `hull_mode::Symbol=:direct`: Hull strategy. `:direct` runs concave hull on a thinned point set; `:stepwise` runs convex hull first, then keeps only points near the hull boundary before concave hull
- `hull_stepwise_max_points::Int=20000`: Maximum points to pass to the stepwise concave hull stage (closest-to-boundary points are kept)
- `hull_stepwise_min_points::Int=5000`: Minimum points to pass to the stepwise concave hull stage (if the distance threshold yields fewer, the closest-to-boundary points are used instead)
- `hull_stepwise_buffer_cells::Real=2.5`: How far from the fast hull boundary (in "grid cell" units) a point must be to be included in the stepwise subset
- `hull_stepwise_passes::Int=2`: Number of stepwise refinement passes (pass 1 uses convex hull; later passes refine using the previously computed hull)
- `show_locations::Bool=false`: Display input locations as colored circular markers
- `location_color::AbstractString="purple"`: Marker color used for the location circles
- `location_size::Number=10`: Marker diameter for the location circles
- `location_names_above::AbstractVector=String[]`: Optional labels plotted above each location marker
- `location_names_below::AbstractVector=String[]`: Optional labels plotted below each location marker
- `show_hull::Bool=false`: Overlay the computed hull polygon for debugging
- `hull_color::AbstractString="magenta"`: Hull trace color when `show_hull=true`
- `hull_line_width::Number=3`: Line width for the hull outline
- `hull_opacity::Real=0.35`: Opacity applied to the hull trace
- `kw...`: Additional keyword arguments passed to the mapbox function

# Debugging
- `progress::Bool=false`: If true, prints stage-by-stage timing/progress information to help diagnose slowdowns
- `progress_every::Int=0`: Row stride for progress logs (0 = auto)

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
	zvalue::AbstractVector{T2};
	preset::Symbol=:balanced,
	zmin::Union{Number,Nothing}=nothing,
	zmax::Union{Number,Nothing}=nothing,
	resolution::Int=_DEFAULT_MAPBOX_CONTOUR_RESOLUTION,
	max_features::Int=_DEFAULT_MAPBOX_CONTOUR_MAX_FEATURES,
	hover_max_features::Int=_DEFAULT_MAPBOX_CONTOUR_HOVER_MAX_FEATURES,
	neighbors::Int=_DEFAULT_MAPBOX_CONTOUR_NEIGHBORS,
	power::Real=2,
	smoothing::Real=0.0,
	contour_levels::Int=20,
	filename::AbstractString="",
	title::AbstractString="",
	title_colorbar::AbstractString=title,
	colorscale::Symbol=:turbo,
	title_length::Number=0,
	opacity::Real=0.7,
	show_locations::Bool=false,
	location_color::AbstractString="purple",
	location_size::Number=10,
	location_names_above::AbstractVector=String[],
	location_names_below::AbstractVector=String[],
	lon_center::Real=minimumnan(lon) + (maximumnan(lon) - minimumnan(lon)) / 2,
	lat_center::Real=minimumnan(lat) + (maximumnan(lat) - minimumnan(lat)) / 2,
	zoom::Number=compute_zoom(lon, lat),
	zoom_fig::Number=zoom,
	style="mapbox://styles/mapbox/satellite-streets-v12",
	mapbox_token=NMFk.mapbox_token,
	figuredir::AbstractString=".",
	format::AbstractString=filename == "" ? "html" : splitext(filename)[end][2:end],
	width::Int=14,
	height::Int=7,
	dpi::Int=200,
	width_pixel::Int=dpi * width,
	height_pixel::Int=dpi * height,
	display_width_pixel::Union{Int,Nothing}=nothing,
	display_height_pixel::Union{Int,Nothing}=nothing,
	scale::Real=1,
	font_size::Number=14,
	font_size_fig::Number=font_size,
	colorbar_bgcolor::AbstractString="#5a5a5a",
	colorbar_font_color::AbstractString="white",
	colorbar_font_family::AbstractString="Arial",
	colorbar_font_bold::Bool=true,
	colorbar_font_size::Number=font_size,
	colorbar_font_size_fig::Number=font_size_fig,
	paper_bgcolor::AbstractString=colorbar_bgcolor,
	hull_vertices::Union{Nothing, Vector{Tuple{Float64, Float64}}}=nothing,
	concave_hull::Bool=true,
	hull_max_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_MAX_POINTS,
	hull_grid_bins::Int=0,
	hull_force_convex_above::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_FORCE_CONVEX_ABOVE,
	hull_concave_max_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_CONCAVE_MAX_POINTS,
	hull_mode::Symbol=_DEFAULT_MAPBOX_CONTOUR_HULL_MODE,
	hull_stepwise_max_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MAX_POINTS,
	hull_stepwise_min_points::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MIN_POINTS,
	hull_stepwise_buffer_cells::Real=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_BUFFER_CELLS,
	hull_stepwise_passes::Int=_DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_PASSES,
	hull_padding::Real=0.0,
	extra_margin::Real=0.0,
	show_hull::Bool=false,
	hull_color::AbstractString="magenta",
	hull_line_width::Number=3,
	hull_opacity::Real=0.35,
	mask_mode::Symbol=_DEFAULT_MAPBOX_CONTOUR_MASK_MODE,
	mask_distance::Union{Nothing, Real}=nothing,
	mask_distance_multiplier::Real=_DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_MULTIPLIER,
	mask_distance_quantile::Real=_DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_QUANTILE,
	mask_distance_max_points::Int=_DEFAULT_MAPBOX_CONTOUR_MASK_DISTANCE_MAX_POINTS,
	quiet::Bool=false,
	progress::Bool=false,
	progress_every::Int=100,
	frame_insufficient_data::Bool=false,
	kw...
) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	start_ns = time_ns()
	last_ns = start_ns
	function _logstep(msg; kwargs...)
		if progress
			now_ns = time_ns()
			elapsed = (now_ns - start_ns) / 1e9
			delta = (now_ns - last_ns) / 1e9
			last_ns = now_ns
			@info "[mapbox_contour] $(msg)" elapsed_s=round(elapsed; digits=2) step_s=round(delta; digits=2) kwargs...
		end
		nothing
	end
	@assert length(lon) == length(lat) == length(zvalue)
	ensure_mapbox_token!(mapbox_token)
	_logstep("Start", n=length(lon), resolution=resolution, preset=String(preset))
	# Presets provide safe defaults for large datasets without requiring many knobs.
	# They only apply when the relevant keyword is still set to its function default.
	if preset == :fast
		(resolution == _DEFAULT_MAPBOX_CONTOUR_RESOLUTION) && (resolution = 30)
		(max_features == _DEFAULT_MAPBOX_CONTOUR_MAX_FEATURES) && (max_features = 10_000)
		(hover_max_features == _DEFAULT_MAPBOX_CONTOUR_HOVER_MAX_FEATURES) && (hover_max_features = 0)
		(neighbors == _DEFAULT_MAPBOX_CONTOUR_NEIGHBORS) && (neighbors = 32)
		(hull_max_points == _DEFAULT_MAPBOX_CONTOUR_HULL_MAX_POINTS) && (hull_max_points = 5_000)
		(hull_force_convex_above == _DEFAULT_MAPBOX_CONTOUR_HULL_FORCE_CONVEX_ABOVE) && (hull_force_convex_above = 1) # always convex for speed
		(hull_concave_max_points == _DEFAULT_MAPBOX_CONTOUR_HULL_CONCAVE_MAX_POINTS) && (hull_concave_max_points = 0)
	elseif preset == :quality
		(resolution == _DEFAULT_MAPBOX_CONTOUR_RESOLUTION) && (resolution = 120)
		(max_features == _DEFAULT_MAPBOX_CONTOUR_MAX_FEATURES) && (max_features = 80_000)
		(neighbors == _DEFAULT_MAPBOX_CONTOUR_NEIGHBORS) && (neighbors = 96)
		(hull_max_points == _DEFAULT_MAPBOX_CONTOUR_HULL_MAX_POINTS) && (hull_max_points = 15_000)
		(hull_force_convex_above == _DEFAULT_MAPBOX_CONTOUR_HULL_FORCE_CONVEX_ABOVE) && (hull_force_convex_above = 0) # allow concave hull; concave input size is capped via hull_concave_max_points
		(hull_concave_max_points == _DEFAULT_MAPBOX_CONTOUR_HULL_CONCAVE_MAX_POINTS) && (hull_concave_max_points = 12_000)
		(hull_mode == _DEFAULT_MAPBOX_CONTOUR_HULL_MODE) && (hull_mode = :stepwise)
		(hull_stepwise_max_points == _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MAX_POINTS) && (hull_stepwise_max_points = 20_000)
		(hull_stepwise_min_points == _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_MIN_POINTS) && (hull_stepwise_min_points = 12_000)
		(hull_stepwise_buffer_cells == _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_BUFFER_CELLS) && (hull_stepwise_buffer_cells = 4.0)
		(hull_stepwise_passes == _DEFAULT_MAPBOX_CONTOUR_HULL_STEPWISE_PASSES) && (hull_stepwise_passes = 2)
		(mask_mode == _DEFAULT_MAPBOX_CONTOUR_MASK_MODE) && (mask_mode = :distance)
	elseif preset == :balanced
		# keep defaults
	else
		throw(ArgumentError("Unknown preset=$(preset). Supported presets: :fast, :balanced, :quality"))
	end
	if resolution < 2
		throw(ArgumentError("resolution must be >= 2"))
	end
	if max_features < 1
		throw(ArgumentError("max_features must be >= 1"))
	end
	if hover_max_features < 0
		throw(ArgumentError("hover_max_features must be >= 0"))
	end
	if hull_max_points < 0
		throw(ArgumentError("hull_max_points must be >= 0"))
	end
	if hull_force_convex_above < 0
		throw(ArgumentError("hull_force_convex_above must be >= 0"))
	end
	if hull_concave_max_points < 0
		throw(ArgumentError("hull_concave_max_points must be >= 0"))
	end
	if hull_stepwise_max_points < 0
		throw(ArgumentError("hull_stepwise_max_points must be >= 0"))
	end
	if hull_stepwise_min_points < 0
		throw(ArgumentError("hull_stepwise_min_points must be >= 0"))
	end
	if hull_stepwise_passes < 1
		throw(ArgumentError("hull_stepwise_passes must be >= 1"))
	end
	if !(hull_mode in (:direct, :stepwise))
		throw(ArgumentError("hull_mode must be :direct or :stepwise"))
	end
	if !(mask_mode in (:hull, :distance, :none, :polygon, :auto))
		throw(ArgumentError("mask_mode must be :hull, :distance, :auto, or :none"))
	end
	if mask_distance_multiplier < 0
		throw(ArgumentError("mask_distance_multiplier must be >= 0"))
	end
	qd = Float64(mask_distance_quantile)
	if !(0.0 <= qd <= 1.0)
		throw(ArgumentError("mask_distance_quantile must be in [0, 1]"))
	end
	if mask_distance_max_points < 0
		throw(ArgumentError("mask_distance_max_points must be >= 0"))
	end
	estimated_features = (resolution - 1) * (resolution - 1)
	if estimated_features > max_features
		new_resolution = max(2, floor(Int, sqrt(max_features)) + 1)
		@warn "Requested contour resolution would generate too many GeoJSON tiles; reducing resolution automatically" resolution=resolution new_resolution=new_resolution max_features=max_features estimated_features=estimated_features
		resolution = new_resolution
	end
	_logstep("Validated settings", resolution=resolution, max_features=max_features, neighbors=neighbors)
	if width_pixel < 500 || height_pixel < 500
		@error "Width and height in pixels should be at least 500 for proper rendering" width_pixel=width_pixel height_pixel=height_pixel
		throw(ArgumentError("Insufficient width/height in pixels"))
	elseif width_pixel > 10000 || height_pixel > 10000
		@warn "Width and height in pixels exceed typical limits; rendering may fail" width_pixel=width_pixel height_pixel=height_pixel
	end
	if title == title_colorbar
		title = ""
	end

	coord_mask = .!isnan.(lon) .& .!isnan.(lat)
	lon_coords = lon[coord_mask]
	lat_coords = lat[coord_mask]

	valid_mask = coord_mask .& .!isnan.(zvalue)
	lon_clean = lon[valid_mask]
	lat_clean = lat[valid_mask]
	values_clean = zvalue[valid_mask]
	_logstep("Filtered inputs", coord_points=length(lon_coords), valid_points=length(lon_clean))

	# Auto masking chooses between support-based masking (:distance) and hull masking (:hull)
	# based on how dense the grid is compared to the sample count.
	if mask_mode == :auto
		@info("Resolving mask_mode=:auto ...")
		ntiles = (resolution - 1) * (resolution - 1)
		npoints = length(lon_clean)
		mask_mode = (ntiles * 100 < npoints) ? :distance : :hull
		@info("Using resolved mask mode $(mask_mode) ...")
		_logstep("Resolved mask_mode=:auto", ntiles=ntiles, npoints=npoints, chosen=String(mask_mode))
	else
		@info("Using specified mask mode $(mask_mode) ...")
		ntiles = (resolution - 1) * (resolution - 1)
		npoints = length(lon_clean)
		_logstep("Using specified mask_mode", ntiles=ntiles, npoints=npoints, mask_mode=String(mask_mode))
	end

	function resolve_location_labels(raw_names, label)
		if isempty(raw_names)
			return nothing
		end
		name_vec = collect(raw_names)
		if length(name_vec) == length(lon)
			return name_vec[valid_mask]
		elseif length(name_vec) == length(lon_clean)
			return name_vec
		else
			@warn "'$(label)' length does not match lon/lat; skipping labels"
			return nothing
		end
	end
	names_above_clean = resolve_location_labels(location_names_above, "location_names_above")
	names_below_clean = resolve_location_labels(location_names_below, "location_names_below")

	if !isempty(lon_clean)
		key_map = Dict{Tuple{Float64, Float64}, Tuple{Float64, Int}}()
		for idx in eachindex(lon_clean)
			key = (Float64(lon_clean[idx]), Float64(lat_clean[idx]))
			val = Float64(values_clean[idx])
			if haskey(key_map, key)
				stored_val, stored_idx = key_map[key]
				if val > stored_val
					key_map[key] = (val, idx)
				end
			else
				key_map[key] = (val, idx)
			end
		end
		indices = sort!([entry[2] for entry in Base.values(key_map)])
		lon_clean = lon_clean[indices]
		lat_clean = lat_clean[indices]
		values_clean = values_clean[indices]
		if names_above_clean !== nothing
			names_above_clean = names_above_clean[indices]
		end
		if names_below_clean !== nothing
			names_below_clean = names_below_clean[indices]
		end
	end

	if names_above_clean !== nothing
		names_above_clean = string.(names_above_clean)
	end
	if names_below_clean !== nothing
		names_below_clean = string.(names_below_clean)
	end

	insufficient_data = length(lon_clean) < 3
	if insufficient_data
		@warn "At least 3 valid data points are required for interpolation; rendering placeholder frame" valid_points=length(lon_clean)
		if frame_insufficient_data
			@info "Rendering empty frame with colorbar only"
		else
			return nothing
		end
	end
	_logstep("Checked minimum data", insufficient_data=insufficient_data)

	need_hull = (mask_mode == :hull || mask_mode == :polygon || show_hull)
	if need_hull && hull_vertices === nothing && concave_hull
		@info("Computing hull for masking/interpolation extent ...")
		# Build the hull from the data actually being interpolated (finite z values).
		# Using all coordinate positions (including NaN-valued samples) can expand the hull
		# and defeat the intended masking.
		use_convex = (hull_force_convex_above > 0) && (length(lon_clean) > hull_force_convex_above)
		thin = (hull_max_points > 0) && (length(lon_clean) > hull_max_points)
		if thin
			_logstep("Thinning points for hull", n_in=length(lon_clean), hull_max_points=hull_max_points)
		end
		try
			hull_vertices = compute_concave_hull_vertices(
				lon_clean,
				lat_clean;
				convex=use_convex,
				mode=hull_mode,
				max_points=hull_max_points,
				grid_bins=hull_grid_bins,
				concave_max_points=hull_concave_max_points,
				stepwise_max_points=hull_stepwise_max_points,
					stepwise_min_points=hull_stepwise_min_points,
				stepwise_buffer_cells=hull_stepwise_buffer_cells,
					stepwise_passes=hull_stepwise_passes,
			)
		catch err
			@warn "Concave hull failed; falling back to convex hull" exception=(err, catch_backtrace())
			hull_vertices = nothing
		end
		if hull_vertices === nothing || length(hull_vertices) < 3
			coords = [(Float64(lon_clean[i]), Float64(lat_clean[i])) for i in eachindex(lon_clean)]
			if length(coords) >= 3
				hull_vertices = compute_convex_hull_vertices(coords)
				if hull_vertices === nothing || length(hull_vertices) < 3
					@info "Hull unavailable; reverting to padded bounding box"
					hull_vertices = nothing
				end
			else
				@info "Hull unavailable; reverting to padded bounding box"
				hull_vertices = nothing
			end
		end
	end
	_logstep("Resolved hull", use_hull=hull_vertices !== nothing)

	mask_tree = nothing
	mask_lon_scale = 1.0
	mask_max_distance = Inf
	if mask_mode == :distance
		if length(lon_clean) >= 2
			lat_mean = _mean_finite(lat_clean; default=0.0)
			mask_lon_scale = cosd(lat_mean)
			coords_mask = Matrix{Float64}(undef, 2, length(lon_clean))
			@inbounds for i in eachindex(lon_clean)
				coords_mask[1, i] = Float64(lon_clean[i]) * mask_lon_scale
				coords_mask[2, i] = Float64(lat_clean[i])
			end
			mask_tree = NearestNeighbors.KDTree(coords_mask)
			spacing = _estimate_nn_spacing(mask_tree, coords_mask; max_points=mask_distance_max_points, q=qd)
			mask_max_distance = mask_distance === nothing ? (Float64(mask_distance_multiplier) * spacing) : Float64(mask_distance)
			_logstep("Distance mask prepared", lon_scale=round(mask_lon_scale; digits=4), spacing=spacing, mask_max_distance=mask_max_distance)
		else
			@warn "Distance mask requested but insufficient points; disabling mask" n=length(lon_clean)
			mask_mode = :none
		end
	end

	lon_source_raw = hull_vertices === nothing ? lon_clean : first.(hull_vertices)
	lat_source_raw = hull_vertices === nothing ? lat_clean : last.(hull_vertices)
	if length(lon_source_raw) == 0
		lon_range_raw = 0.0
		lat_range_raw = 0.0
	else
		lon_range_raw = maximum(lon_source_raw) - minimum(lon_source_raw)
		lat_range_raw = maximum(lat_source_raw) - minimum(lat_source_raw)
	end
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
		if _polygon_self_intersects(hull_vertices)
			@warn "Expanded concave hull self-intersection detected; reverting to convex hull"
			coords = [(Float64(lon_clean[i]), Float64(lat_clean[i])) for i in eachindex(lon_clean)]
			if length(coords) >= 3
				hull_vertices = compute_convex_hull_vertices(coords)
				if hull_vertices !== nothing && (effective_padding > 0 || effective_margin > 0)
					hull_vertices = _expand_polygon_vertices(
						hull_vertices;
						lon_span=lon_range_raw,
						lat_span=lat_range_raw,
						padding_fraction=effective_padding,
						margin=effective_margin
					)
					if _polygon_self_intersects(hull_vertices)
						@warn "Expanded convex hull still self-intersects; using bounding box"
						hull_vertices = nothing
					end
				end
			else
				@warn "Insufficient coordinates for convex hull fallback; using bounding box"
				hull_vertices = nothing
			end
		end
	end
	lon_source_raw = hull_vertices === nothing ? lon_clean : first.(hull_vertices)
	lat_source_raw = hull_vertices === nothing ? lat_clean : last.(hull_vertices)
	if length(lon_source_raw) > 0
		lon_range_raw = maximum(lon_source_raw) - minimum(lon_source_raw)
		lat_range_raw = maximum(lat_source_raw) - minimum(lat_source_raw)
	end
	hull_plot_vertices = hull_vertices === nothing ? nothing : copy(hull_vertices)

	lon_source = lon_source_raw
	lat_source = lat_source_raw
	lon_range = lon_range_raw
	lat_range = lat_range_raw
	lon_span = lon_range == 0 ? 1e-6 : lon_range
	lat_span = lat_range == 0 ? 1e-6 : lat_range
	padding = hull_vertices === nothing ? 0.1 : 0.0
	margin = hull_vertices === nothing ? effective_margin : 0.0
	if length(lon_source) > 0
		lon_min = minimum(lon_source) - lon_span * padding - margin
		lon_max = maximum(lon_source) + lon_span * padding + margin
		lat_min = minimum(lat_source) - lat_span * padding - margin
		lat_max = maximum(lat_source) + lat_span * padding + margin
	end

	traces = PlotlyJS.GenericTrace{Dict{Symbol, Any}}[]
	if insufficient_data
		zmin_target, zmax_target = _resolve_color_bounds(zmin, zmax, values_clean)
		colorbar_ticks = _colorbar_tick_values(zmin_target, zmax_target)
		colorbar_tick_labels = _colorbar_tick_labels(colorbar_ticks)
		colorbar_attr = mapbox_colorbar_attr(title_colorbar, title_length; font_size=colorbar_font_size, font_color=colorbar_font_color, font_family=colorbar_font_family, bgcolor=colorbar_bgcolor, bold=colorbar_font_bold, tickmode="array", tickvals=colorbar_ticks, ticktext=colorbar_tick_labels)
		dummy_marker = PlotlyJS.attr(
			size=1,
			opacity=0.0,
			color=[zmin_target, zmax_target],
			colorscale=NMFk.colorscale(colorscale),
			showscale=true,
			colorbar=colorbar_attr,
			cmin=zmin_target,
			cmax=zmax_target
		)
		dummy_trace = PlotlyJS.scattermapbox(
			lon=[lon_center, lon_center],
			lat=[lat_center, lat_center],
			mode="markers",
			marker=dummy_marker,
			name="No Data",
			hoverinfo="skip",
			showlegend=false
		)
		push!(traces, dummy_trace)
	else
		_logstep("Building interpolation grid")
		lon_grid = range(lon_min, lon_max, length=resolution)
		lat_grid = range(lat_min, lat_max, length=resolution)

		z_grid = Matrix{Float64}(undef, resolution, resolution)
		npoints = length(lon_clean)
		use_knn = (neighbors > 0) || (neighbors == 0 && npoints >= 2_000)
		k = neighbors > 0 ? neighbors : 32
		_logstep("Interpolating", method=(use_knn ? "knn-idw" : "bruteforce-idw"), npoints=npoints, k=k, grid=resolution*resolution)
		if use_knn && npoints > 0
			# KNN-based IDW avoids O(npoints) work per grid cell for large datasets.
			coords = Matrix{Float64}(undef, 2, npoints)
			@inbounds begin
				coords[1, :] = lon_clean
				coords[2, :] = lat_clean
			end
			@info("KNN-based interpolation ...")
			tree = NearestNeighbors.KDTree(coords)
			k_eff = min(max(1, k), npoints)
			q = Vector{Float64}(undef, 2)
			stride = progress_every > 0 ? progress_every : max(1, cld(resolution, 10))
			!quiet && @info("Performing IDW interpolation on grid ...")
			@inbounds for (i, lat_interp) in enumerate(lat_grid)
				if progress && (i == 1 || i % stride == 0 || i == resolution)
					_logstep("interpolation progress", row=i, total_rows=resolution, pct=round(100 * i / resolution; digits=1))
				end
				q[2] = Float64(lat_interp)
				for (j, lon_interp) in enumerate(lon_grid)
					q[1] = Float64(lon_interp)
					idxs, d2s = NearestNeighbors.knn(tree, q, k_eff, true)
					z_val = 0.0
					w_sum = 0.0
					assigned = false
					for t in eachindex(idxs)
						d2 = d2s[t]
						if d2 <= 1e-20
							z_grid[i, j] = Float64(values_clean[idxs[t]])
							assigned = true
							break
						end
						d = sqrt(d2)
						w = 1.0 / (d^power + smoothing)
						z_val += w * Float64(values_clean[idxs[t]])
						w_sum += w
					end
					if !assigned
						z_grid[i, j] = w_sum > 0 ? (z_val / w_sum) : NaN
					end
				end
			end
		else
			!quiet && @info("Brute-force interpolation ...") # Brute-force fallback; fine for small point sets.
			stride = progress_every > 0 ? progress_every : max(1, cld(resolution, 10))
			@inbounds for (i, lat_interp) in enumerate(lat_grid)
				if progress && (i == 1 || i % stride == 0 || i == resolution)
					_logstep("Interpolation progress", row=i, total_rows=resolution, pct=round(100 * i / resolution; digits=1))
				end
				for (j, lon_interp) in enumerate(lon_grid)
					distances = sqrt.((lon_clean .- lon_interp).^2 .+ (lat_clean .- lat_interp).^2)
					min_dist = minimum(distances)
					if min_dist < 1e-10
						closest_idx = Base.findfirst(distances .== min_dist)
						z_grid[i, j] = values_clean[closest_idx]
					else
						weights = 1.0 ./ (distances.^power .+ smoothing)
						z_grid[i, j] = sum(weights .* values_clean) / sum(weights)
					end
				end
			end
		end
		if mask_mode == :distance && mask_tree !== nothing && length(lon_grid) >= 2 && length(lat_grid) >= 2 && isfinite(mask_max_distance)
			lon_step = abs(Float64(lon_grid[2]) - Float64(lon_grid[1])) * mask_lon_scale
			lat_step = abs(Float64(lat_grid[2]) - Float64(lat_grid[1]))
			min_keep = 0.75 * hypot(lon_step, lat_step)
			if mask_max_distance < min_keep
				mask_max_distance = min_keep
				_logstep("Distance mask increased for grid", min_keep=min_keep, mask_max_distance=mask_max_distance)
			end
		end
		_logstep("Building GeoJSON")
		geojson_tiles, geojson_ids, geojson_values = _build_geojson_tiles(
			lon_grid,
			lat_grid,
			z_grid;
			mask_polygon=(mask_mode in (:hull, :polygon) ? hull_vertices : nothing),
			mask_mode=mask_mode,
			mask_tree=mask_tree,
			mask_max_distance=mask_max_distance,
			mask_lon_scale=mask_lon_scale,
			progress=progress,
			progress_every=progress_every,
		)
		_logstep("GeoJSON built", tiles=length(geojson_ids))
		disable_hover = hover_max_features == 0 ? true : (length(geojson_ids) > hover_max_features)
		if disable_hover
			_logstep("Disabling hover (too many features)", tiles=length(geojson_ids), hover_max_features=hover_max_features)
		end
		zmin_target, zmax_target = _resolve_color_bounds(zmin, zmax, geojson_values)
		_logstep("Resolved color bounds", zmin=zmin_target, zmax=zmax_target)
		colorbar_ticks = _colorbar_tick_values(zmin_target, zmax_target)
		colorbar_tick_labels = _colorbar_tick_labels(colorbar_ticks)
		colorbar_attr = mapbox_colorbar_attr(title_colorbar, title_length; font_size=colorbar_font_size, font_color=colorbar_font_color, font_family=colorbar_font_family, bgcolor=colorbar_bgcolor, bold=colorbar_font_bold, tickmode="array", tickvals=colorbar_ticks, ticktext=colorbar_tick_labels)

		if !isempty(geojson_ids)
			_logstep("Building choropleth trace")
			plot_values = copy(geojson_values)
			plot_values .= clamp.(plot_values, zmin_target, zmax_target)
			if contour_levels > 1 && zmax_target > zmin_target
				step = (zmax_target - zmin_target) / contour_levels
				if step > 0
					for idx in eachindex(plot_values)
						level = zmin_target + round((plot_values[idx] - zmin_target) / step) * step
						plot_values[idx] = clamp(level, zmin_target, zmax_target)
					end
				end
			end
			choropleth_trace = PlotlyJS.choroplethmapbox(
				geojson=geojson_tiles,
				locations=geojson_ids,
				z=plot_values,
				customdata=geojson_values,
				featureidkey="id",
				colorscale=NMFk.colorscale(colorscale),
				opacity=opacity,
				zmin=zmin_target,
				zmax=zmax_target,
				marker=PlotlyJS.attr(line=PlotlyJS.attr(width=0)),
				colorbar=colorbar_attr,
				hoverinfo=disable_hover ? "skip" : "",
				hovertemplate=disable_hover ? "<extra></extra>" : "<b>Value:</b> %{customdata:.3f}<extra></extra>",
				name="Interpolated Field",
				showscale=true,
				showlegend=false
			)
			push!(traces, choropleth_trace)
		else
			_logstep("GeoJSON empty; building fallback scatter")
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
						color=clamp.(fallback_vals, zmin_target, zmax_target),
						colorscale=NMFk.colorscale(colorscale),
						cmin=zmin_target,
						cmax=zmax_target,
						opacity=opacity,
						colorbar=colorbar_attr
					),
					hoverinfo=disable_hover ? "skip" : "",
					hovertemplate=disable_hover ? "<extra></extra>" : "<b>Lon:</b> %{lon}<br><b>Lat:</b> %{lat}<br><b>Value:</b> %{marker.color}<extra></extra>",
					showlegend=false,
					name="Interpolated Surface"
				)
				push!(traces, contour_trace)
			end
		end
	end
	if show_locations && !isempty(lon_clean) && length(lon_clean) < 1000
		!quiet && @info("Plotting location markers")
		marker_attr = PlotlyJS.attr(
			size=location_size,
			color=location_color,
			opacity=0.9,
			line=PlotlyJS.attr(color=location_color, width=0)
		)
		location_trace = PlotlyJS.scattermapbox(
			lon=lon_clean,
			lat=lat_clean,
			mode="markers",
			marker=marker_attr,
			name="Locations",
			hovertemplate="<b>Lon:</b> %{lon:.4f}<br><b>Lat:</b> %{lat:.4f}<extra></extra>",
			showlegend=false
		)
		push!(traces, location_trace)

		label_font = PlotlyJS.attr(color=location_color, size=max(8, Int(round(font_size - 2))))
		if names_above_clean !== nothing
			labels_above_trace = PlotlyJS.scattermapbox(
				lon=lon_clean,
				lat=lat_clean,
				mode="text",
				text=names_above_clean,
				textposition="top center",
				textfont=label_font,
				hoverinfo="skip",
				showlegend=false,
				name="Location Labels (Top)"
			)
			push!(traces, labels_above_trace)
		end
		if names_below_clean !== nothing
			labels_below_trace = PlotlyJS.scattermapbox(
				lon=lon_clean,
				lat=lat_clean,
				mode="text",
				text=names_below_clean,
				textposition="bottom center",
				textfont=label_font,
				hoverinfo="skip",
				showlegend=false,
				name="Location Labels (Bottom)"
			)
			push!(traces, labels_below_trace)
		end
	end

	if show_hull && hull_plot_vertices !== nothing
		!quiet && @info("Plotting hull polygon ...")
		hull_lon = first.(hull_plot_vertices)
		hull_lat = last.(hull_plot_vertices)
		if isempty(hull_lon)
			@warn "Hull visualization requested but polygon is empty"
		else
			if hull_lon[end] != hull_lon[1] || hull_lat[end] != hull_lat[1]
				push!(hull_lon, hull_lon[1])
				push!(hull_lat, hull_lat[1])
			end
			hull_trace = PlotlyJS.scattermapbox(
				lon=hull_lon,
				lat=hull_lat,
				mode="lines",
				line=PlotlyJS.attr(color=hull_color, width=hull_line_width),
				opacity=hull_opacity,
				name="Mask Hull",
				showlegend=false
			)
			push!(traces, hull_trace)
		end
	end

	_logstep("Building layout")
	layout = plotly_layout(
		lon_center, lat_center, zoom;
		width=display_width_pixel,
		height=display_height_pixel,
		title=title,
		font_size=font_size,
		paper_bgcolor=paper_bgcolor,
		style=style,
		mapbox_token=mapbox_token
	)
	responsive = isnothing(display_width_pixel) && isnothing(display_height_pixel)
	_logstep("Rendering Plotly figure", responsive=responsive)
	p = PlotlyJS.plot(traces, layout; config=PlotlyJS.PlotConfig(scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=responsive))
	_logstep("Render complete")
	!quiet && display(p)

	if filename != ""
		colorbar_attr_fig = mapbox_colorbar_attr(title_colorbar, title_length; font_size=colorbar_font_size_fig, font_color=colorbar_font_color, font_family=colorbar_font_family, bgcolor=colorbar_bgcolor, bold=colorbar_font_bold, tickmode="array", tickvals=colorbar_ticks, ticktext=colorbar_tick_labels)

		traces_fig = deepcopy(traces)
		for tr in traces_fig
			if haskey(tr.fields, :marker)
				marker = tr.fields[:marker]
				if marker isa AbstractDict && haskey(marker, :colorbar)
					marker[:colorbar] = colorbar_attr_fig
					tr.fields[:marker] = marker
				end
			end
			if haskey(tr.fields, :colorbar)
				tr.fields[:colorbar] = colorbar_attr_fig
			end
		end
		layout = plotly_layout(
			lon_center, lat_center, zoom_fig;
			width=width_pixel,
			height=height_pixel,
			title=title,
			font_size=font_size_fig,
			paper_bgcolor=paper_bgcolor,
			style=style,
			mapbox_token=mapbox_token
		)
		p = PlotlyJS.plot(traces_fig, layout; config=PlotlyJS.PlotConfig(scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		safe_savefig(p, fn; format=format, width=width_pixel, height=height_pixel, scale=scale)
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
		closest_idx = Base.findfirst(distances .== min_dist)
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