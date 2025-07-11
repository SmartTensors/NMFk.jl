import VegaLite
import VegaDatasets
import DataFrames
import Mads
import PlotlyJS
import Interpolations
import NearestNeighbors

mapbox_token = "pk.eyJ1IjoibW9udHl2IiwiYSI6ImNsMDhvNTJwMzA1OHgzY256N2c2aDdzdXoifQ.cGUz0Wuc3rYRqGNwm9v5iQ"

regex_lon = r"^[Xx]$|^[Ll]on$|^LONGITUDE$|^LON$|^[Ll]ongitude$" # regex for longitude
regex_lat = r"^[Yy]$|^[Ll]at$|^LATITUDE$|^LAT$|^[Ll]atitude$" # regex for latitude

# Plot a county based (FIPS) heatmap
function plotmap(W::AbstractMatrix, H::AbstractMatrix, fips::AbstractVector, dim::Integer=1; casefilename::AbstractString="", figuredir::AbstractString=".", moviedir::AbstractString=".", dates=nothing, plotseries::Bool=true, plotpeaks::Bool=false, plottransients::Bool=false, quiet::Bool=false, movie::Bool=false, hsize::Measures.AbsoluteLength=12Compose.inch, vsize::Measures.AbsoluteLength=6Compose.inch, dpi::Integer=300, name::AbstractString="Wave peak", cleanup::Bool=true, vspeed::Number=1.0, kw...)
	@assert size(W, 2) == size(H, 1)
	Wa, _, _ = normalizematrix_col!(W)
	Ha, _, _ = normalizematrix_row!(H)
	recursivemkdir(figuredir; filename=false)
	if dim == 1
		odim = 2
		Ma = Wa
		S = W
		Fa = Ha
	else
		odim = 1
		Ma = Ha
		S = H
		Fa = Wa
	end
	signalorderassignments, signalpeakindex = NMFk.signalorderassignments(Ma, odim)
	nt = dim == 1 ? (Colon(), signalorderassignments) : (signalorderassignments, Colon())
	if !isnothing(dates)
		@assert length(dates) == size(Ma, 1)
		ndates = dates[signalpeakindex]
	else
		ndates = dates
	end
	if plotseries
		fn = casefilename == "" ? "" : joinpathcheck(figuredir, casefilename * "-waves.png")
		Mads.plotseries(S[nt...] ./ maximum(S), fn; xaxis=dates, names=["$name $(ndates[i])" for i in signalorderassignments])
		if movie && casefilename != ""
			color = Mads.plotseries(S[nt...] ./ maximum(S); xaxis=dates, names=["S$i $(ndates[k])" for (i, k) in enumerate(signalorderassignments)], code=true, quiet=true)
			progressbar = NMFk.make_progressbar_2d(color)
			for i in eachindex(dates)
				p = progressbar(i, true, 1, dates[1])
				Gadfly.draw(Gadfly.PNG(joinpathcheck(moviedir, casefilename * "-progressbar-$(lpad(i, 6, '0')).png"), hsize, vsize; dpi=dpi), p)
				!quiet && (Mads.display(p; gw=hsize, gh=vsize))
			end
			makemovie(; moviedir=moviedir, prefix=casefilename * "-progressbar", keyword="", numberofdigits=6, cleanup=cleanup, vspeed=vspeed)
		end
	end
	if plotpeaks
		NMFk.plotmap(Fa, fips, dim, signalorderassignments; dates=ndates, figuredir=figuredir, casefilename=casefilename, quiet=quiet, kw...)
	end
	if plottransients
		for (i, k) in enumerate(signalorderassignments)
			Xe = dim == 1 ? W[:, k:k] * H[k:k, :] : permutedims(W[:, k:k] * H[k:k, :])
			# p = signalpeakindex[k]
			# NMFk.plotmap(Xe[p:p,:], fips; dates=[ndates[k]], figuredir=moviedir, casefilename=casefilename * "-signal-$(i)", datetext="S$(i) ", movie=movie, quiet=!movie, kw...)
			NMFk.plotmap(Xe, fips; dates=dates, figuredir=moviedir, casefilename=casefilename * "-signal-$(i)", datetext="S$(i) ", movie=movie, quiet=!movie, cleanup=cleanup, vspeed=vspeed, kw...)
		end
	end
	return nothing
end

# Plot a county based (FIPS) heatmap
function plotmap(X::AbstractMatrix, fips::AbstractVector, dim::Integer=1, signalorderassignments::AbstractVector=axes(X, dim); signalid::AbstractVector=axes(X, dim), us10m=VegaDatasets.dataset("us-10m"), goodcounties::AbstractVector=trues(length(fips)), dates=nothing, casefilename::AbstractString="", figuredir::AbstractString=".", title::Bool=false, datetext::AbstractString="", titletext::AbstractString="", leadingzeros::Integer=1 + convert(Int64, ceil(log10(length(signalorderassignments)))), scheme::AbstractString="redyellowgreen", zmin::Number=0, zmax::Number=1, zformat="f", quiet::Bool=false, movie::Bool=false, cleanup::Bool=true, vspeed::Number=1.0)
	odim = dim == 1 ? 2 : 1
	@assert size(X, odim) == length(fips[goodcounties])
	@assert length(signalorderassignments) == length(signalid)
	if !isnothing(dates)
		@assert size(X, dim) == length(dates)
	end
	recursivemkdir(figuredir; filename=false)
	df = DataFrames.DataFrame(; FIPS=[fips[goodcounties]; fips[.!goodcounties]])
	for (i, k) in enumerate(signalorderassignments)
		nt = ntuple(j -> (j == dim ? k : Colon()), ndims(X))
		df[!, :Z] = [vec(X[nt...]); zeros(sum(.!goodcounties))]
		signalidtext = eltype(signalid) <: Integer ? lpad(signalid[i], leadingzeros, '0') : signalid[i]
		if title || (!isnothing(dates) && titletext != "")
			ttitle = "$(titletext) $(signalidtext)"
			if !isnothing(dates)
				ttitle *= ": $(datetext): $(dates[k])"
			end
			ltitle = ""
		else
			ttitle = nothing
			if !isnothing(dates)
				ltitle = datetext .* "$(dates[k])"
			else
				ltitle = "$(titletext) $(signalidtext)"
			end
		end
		p = VegaLite.@vlplot(
			title = ttitle,
			:geoshape,
			width = 500, height = 300,
			data = {
				values = us10m,
				format = {
					type = :topojson,
					feature = :counties
				}
			},
			transform = [{
				lookup = :id,
				from = {
					data = df,
					key = :FIPS,
					fields = ["Z"]
				}
			}],
			projection = {type = :albersUsa},
			color = {title = ltitle, field = "Z", type = "quantitative", scale = {scheme = scheme, clamp = true, reverse = true, domain = [zmin, zmax]}, legend = {format = zformat}}
		)
		!quiet && (display(p); println())
		if casefilename != ""
			VegaLite.save(joinpathcheck("$(figuredir)", "$(casefilename)-$(signalidtext).png"), p)
		end
	end
	if casefilename != "" && movie
		makemovie(; moviedir=figuredir, prefix=casefilename, keyword="", numberofdigits=leadingzeros, cleanup=cleanup, vspeed=vspeed)
	end
	return nothing
end

# Plot a county based (FIPS) heatmap
function plotmap(X::AbstractVector, fips::AbstractVector; us10m=VegaDatasets.dataset("us-10m"), goodcounties::AbstractVector=trues(length(fips)), casefilename::AbstractString="", figuredir::AbstractString=".", title::AbstractString="", quiet::Bool=false, scheme::AbstractString="category10", zmin::Number=0, zmax::Number=1)
	recursivemkdir(figuredir; filename=false)
	@assert length(X) == length(fips)
	nc = length(unique(sort(X))) + 1
	df = DataFrames.DataFrame(; FIPS=[fips[goodcounties]; fips[.!goodcounties]], Z=[X; zeros(sum(.!goodcounties))])
	p = VegaLite.@vlplot(
		:geoshape,
		width = 500, height = 300,
		data = {
			values = us10m,
			format = {
				type = :topojson,
				feature = :counties
			}
		},
		transform = [{
			lookup = :id,
			from = {
				data = df,
				key = :FIPS,
				fields = ["Z"]
			}
		}],
		projection = {type = :albersUsa},
		color = {title = title, field = "Z", type = "ordinal", scale = {scheme = vec("#" .* Colors.hex.(parse.(Colors.Colorant, NMFk.colors), :RGB))[1:nc], reverse = true, domainMax = zmax, domainMin = zmin}}
	)
	if casefilename != ""
		VegaLite.save(joinpathcheck("$(figuredir)", "$(casefilename).png"), p)
	end
	!quiet && (display(p); println())
	return p
end

function plotmap(df::DataFrames.DataFrame; namesmap::AbstractVector=names(df), filename::AbstractString="", kw...)
	lon, lat = get_lonlat(df)
	if isnothing(lon) || isnothing(lat)
		@error("Longitude and latitude columns are required for plotting!")
		return nothing
	end
	fileroot, fileext = splitext(filename)
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
			p = plotmap(lon, lat, df[!, a]; filename=f, title=varname, kw...)
			display(p)
		else
			if length(namesmap) == length(names(df))
				col += 1
			end
		end
	end
	return nothing
end

# Plot a scatter geo map (continuous color scale)
function plotmap(lon::AbstractVector{T1}, lat::AbstractVector{T1}, color::AbstractVector{T2}; figuredir::AbstractString=".", filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end], title::AbstractString="", text::AbstractVector=repeat([""], length(lon)), scope::AbstractString="usa", projection_type::AbstractString="albers usa", marker_size::Number=5, marker_size_fig::Number=10, font_size::Number=14, font_size_fig::Number=46, width::Int=2800, height::Int=1400, scale::Real=1, showland::Bool=true, kw...) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	geo = PlotlyJS.attr(;
		scope=scope,
		projection_type=projection_type,
		showland=showland,
		landcolor="rgb(217, 217, 217)",
		subunitwidth=1,
		countrywidth=1,
		subunitcolor="rgb(255,255,255)",
		countrycolor="rgb(255,255,255)")
	function layout_fig(font_size::Number)
		layout = PlotlyJS.Layout(;
		margin=PlotlyJS.attr(; r=0, t=0, b=0, l=0),
		mapbox=PlotlyJS.attr(; accesstoken=mapbox_token, style="mapbox://styles/mapbox/satellite-streets-v12"),
		title=PlotlyJS.attr(;
			text=title,
			font=PlotlyJS.attr(; size=font_size, color="black"),
			x=0.5, y=0.95, xanchor="center", yanchor="bottom",
			_pad=PlotlyJS.attr(; t=10)),
		showlegend=false,
		geo=geo)
		return layout
	end
	function trace_fig(marker_size::Number, font_size::Number)
		trace = PlotlyJS.scattergeo(;
			locationmode="USA-states",
			lon=lon,
			lat=lat,
			hoverinfo="text",
			text=text,
			marker=PlotlyJS.attr(; size=marker_size, color=color, colorscale=NMFk.colorscale(:rainbow), colorbar=PlotlyJS.attr(; thickness=20, len=0.5, width=100, tickfont_size=font_size), line_width=0, line_color="black"))
		return trace
	end
	if filename != ""
		p_fig = PlotlyJS.plot(trace_fig(marker_size_fig, font_size_fig), layout_fig(font_size_fig))
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p_fig, fn; format=format, width=width, height=height, scale=scale)
	end
	p = PlotlyJS.plot(trace_fig(marker_size, font_size), layout_fig(font_size))
	return p
end

# Plot a scatter geo map (categorical)
function plotmap(lon::AbstractVector{T1}, lat::AbstractVector{T1}, color::AbstractVector{T2}; figuredir::AbstractString=".", filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end], title::AbstractString="", text::AbstractVector=repeat([""], length(lon)), scope::AbstractString="usa", projection_type::AbstractString="albers usa", marker_size::Number=10, showland::Bool=true, kw...) where {T1 <: AbstractFloat, T2 <: Union{Integer, AbstractString, AbstractChar}}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	traces = Vector{PlotlyJS.GenericTrace{Dict{Symbol, Any}}}(undef, 0)
	for (j, i) in enumerate(unique(sort(color)))
		iz = color .== i
		jj = j % length(NMFk.colors)
		k = jj == 0 ? length(NMFk.colors) : jj
		trace = PlotlyJS.scattergeo(;
			locationmode="USA-states",
			lon=lon[iz],
			lat=lat[iz],
			hoverinfo="text",
			text=text[iz],
			name="$i $(sum(iz))",
			marker=PlotlyJS.attr(; size=marker_size, color=NMFk.colors[k]))
		push!(traces, trace)
	end
	geo = PlotlyJS.attr(;
		scope=scope,
		projection_type=projection_type,
		showland=showland,
		landcolor="rgb(217, 217, 217)",
		subunitwidth=1,
		countrywidth=1,
		subunitcolor="rgb(255,255,255)",
		countrycolor="rgb(255,255,255)")
	layout = PlotlyJS.Layout(; title=title, geo=geo)
	p = PlotlyJS.plot(traces, layout)
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	return p
end

function get_lonlat(df::DataFrames.DataFrame)
	rlon = occursin.(regex_lon, names(df))
	rlat = occursin.(regex_lat, names(df))
	if any(rlon) && any(rlat)
		lon = df[!, first(names(df)[rlon])]
		lat = df[!, first(names(df)[rlat])]
		return lon, lat
	else
		@error("No longitude or latitude column found in the dataframe!")
		return nothing, nothing
	end
end

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
			color=color,
			colorscale=NMFk.colorscale(colorscale),
			cmin=zmin,
			cmax=zmax,
			colorbar=colorbar_attr
		)
		plot = PlotlyJS.scattermapbox(;
			lon=lon,
			lat=lat,
			text=text,
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
		color=color,
		cmin=zmin,
		cmax=zmax,
		colorscale=NMFk.colorscale(colorscale),
		colorbar=colorbar_attr
	)
	plot = PlotlyJS.scattermapbox(;
		lon=lon,
		lat=lat,
		text=text,
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
	font_size::Number=46,
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
function plotly_layout(lonc::Number=-105.9378, latc::Number=35.6870, zoom::Number=4; paper_bgcolor::AbstractString="white", width::Int=2800, height::Int=1400, title::AbstractString="", font_size::Number=46, font_color="black", style="mapbox://styles/mapbox/satellite-streets-v12", mapbox_token=NMFk.mapbox_token)
	layout = PlotlyJS.Layout(;
		margin=PlotlyJS.attr(; r=0, t=0, b=0, l=0),
		title=PlotlyJS.attr(;
			text=title,
			font=PlotlyJS.attr(; size=font_size, color=font_color),
			x=0.5, y=0.95,
			xanchor="center", yanchor="bottom",
			_pad=PlotlyJS.attr(; t=10)),
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