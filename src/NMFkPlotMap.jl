import VegaLite
import VegaDatasets
import DataFrames
import Mads
import PlotlyJS

mapbox_token = "pk.eyJ1IjoibW9udHl2IiwiYSI6ImNsMDhvNTJwMzA1OHgzY256N2c2aDdzdXoifQ.cGUz0Wuc3rYRqGNwm9v5iQ"

# Plot a county based (FIPS) heatmap
function plotmap(W::AbstractMatrix, H::AbstractMatrix, fips::AbstractVector, dim::Integer=1; casefilename::AbstractString="", figuredir::AbstractString=".", moviedir::AbstractString=".", dates=nothing, plotseries::Bool=true, plotpeaks::Bool=false, plottransients::Bool=false, quiet::Bool=false, movie::Bool=false, hsize::Measures.AbsoluteLength=12Compose.inch, vsize::Measures.AbsoluteLength=3Compose.inch, dpi::Integer=150, name::AbstractString="Wave peak", cleanup::Bool=true, vspeed::Number=1.0, kw...)
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
	nt = dim == 1 ? (Colon(),signalorderassignments) : (signalorderassignments,Colon())
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
			color = Mads.plotseries(S[nt...] ./ maximum(S); xaxis=dates, names=["S$i $(ndates[k])" for (i,k) in enumerate(signalorderassignments)], code=true, quiet=true)
			progressbar = NMFk.make_progressbar_2d(color)
			for i = eachindex(dates)
				p = progressbar(i, true, 1, dates[1])
				Gadfly.draw(Gadfly.PNG(joinpathcheck(moviedir, casefilename * "-progressbar-$(lpad(i, 6, '0')).png"), hsize, vsize, dpi=dpi), p)
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
			Xe = dim == 1 ? W[:,k:k] * H[k:k,:] : permutedims(W[:,k:k] * H[k:k,:])
			# p = signalpeakindex[k]
			# NMFk.plotmap(Xe[p:p,:], fips; dates=[ndates[k]], figuredir=moviedir, casefilename=casefilename * "-signal-$(i)", datetext="S$(i) ", movie=movie, quiet=!movie, kw...)
			NMFk.plotmap(Xe, fips; dates=dates, figuredir=moviedir, casefilename=casefilename * "-signal-$(i)", datetext="S$(i) ", movie=movie, quiet=!movie, cleanup=cleanup, vspeed=vspeed, kw...)
		end
	end
end

# Plot a county based (FIPS) heatmap
function plotmap(X::AbstractMatrix, fips::AbstractVector, dim::Integer=1, signalorderassignments::AbstractVector=1:size(X, dim); signalid::AbstractVector=1:size(X, dim), us10m=VegaDatasets.dataset("us-10m"), goodcounties::AbstractVector=trues(length(fips)), dates=nothing, casefilename::AbstractString="", figuredir::AbstractString=".", title::Bool=false, datetext::AbstractString="", titletext::AbstractString="", leadingzeros::Integer=1 + convert(Int64, ceil(log10(length(signalorderassignments)))), scheme::AbstractString="redyellowgreen", zmin::Number=0, zmax::Number=1, zformat="f", quiet::Bool=false, movie::Bool=false, cleanup::Bool=true, vspeed::Number=1.0)
	odim = dim == 1 ? 2 : 1
	@assert size(X, odim) == length(fips[goodcounties])
	@assert length(signalorderassignments) == length(signalid)
	if !isnothing(dates)
		@assert size(X, dim) == length(dates)
	end
	recursivemkdir(figuredir; filename=false)
	df = DataFrames.DataFrame(FIPS=[fips[goodcounties]; fips[.!goodcounties]])
	for (i, k) in enumerate(signalorderassignments)
		nt = ntuple(j->(j == dim ? k : Colon()), ndims(X))
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
			title=ttitle,
			:geoshape,
			width=500, height=300,
			data={
				values=us10m,
				format={
					type=:topojson,
					feature=:counties
				}
			},
			transform=[{
				lookup=:id,
				from={
					data=df,
					key=:FIPS,
					fields=["Z"]
				}
			}],
			projection={type=:albersUsa},
			color={title=ltitle, field="Z", type="quantitative", scale={scheme=scheme, clamp=true, reverse=true, domain=[zmin, zmax]}, legend={format=zformat}}
		)
		!quiet && (display(p); println())
		if casefilename != ""
			VegaLite.save(joinpathcheck("$(figuredir)", "$(casefilename)-$(signalidtext).png"), p)
		end
	end
	if casefilename != "" && movie
		makemovie(; moviedir=figuredir, prefix=casefilename, keyword="", numberofdigits=leadingzeros, cleanup=cleanup, vspeed=vspeed)
	end
end

# Plot a county based (FIPS) heatmap
function plotmap(X::AbstractVector, fips::AbstractVector; us10m=VegaDatasets.dataset("us-10m"), goodcounties::AbstractVector=trues(length(fips)), casefilename::AbstractString="", figuredir::AbstractString=".", title::AbstractString="", quiet::Bool=false, scheme::AbstractString="category10", zmin::Number=0, zmax::Number=1)
	recursivemkdir(figuredir; filename=false)
	@assert length(X) == length(fips)
	nc = length(unique(sort(X))) + 1
	df = DataFrames.DataFrame(FIPS=[fips[goodcounties]; fips[.!goodcounties]], Z=[X; zeros(sum(.!goodcounties))])
	p = VegaLite.@vlplot(
		:geoshape,
		width=500, height=300,
		data={
			values=us10m,
			format={
				type=:topojson,
				feature=:counties
			}
		},
		transform=[{
			lookup=:id,
			from={
				data=df,
				key=:FIPS,
				fields=["Z"]
			}
		}],
		projection={type=:albersUsa},
		color={title=title, field="Z", type="ordinal", scale={scheme=vec("#" .* Colors.hex.(parse.(Colors.Colorant, NMFk.colors), :RGB))[1:nc], reverse=true, domainMax=zmax, domainMin=zmin}}
	)
	!quiet && (display(p); println())
	if casefilename != ""
		VegaLite.save(joinpathcheck("$(figuredir)", "$(casefilename).png"), p)
	end
end

function plotmap(df::DataFrames.DataFrame; kw...)
	plotmap(df.Lon, df.Lat, df[!, 3]; kw...)
end

# Plot a scatter geo map (continuous color scale)
function plotmap(lon::AbstractVector{T1}, lat::AbstractVector{T1}, color::AbstractVector{T2}; figuredir::AbstractString=".", filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end], title::AbstractString="", text::AbstractVector=repeat([""], length(lon)), scope::AbstractString="usa", projection_type::AbstractString="albers usa", size::Number=5, showland::Bool=true, kw...) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	trace = PlotlyJS.scattergeo(;
		locationmode="USA-states",
		lon=lon
,
		lat=lat,
		hoverinfo="text",
		text=text,
		marker=PlotlyJS.attr(; size=size, color=color, colorscale=NMFk.colorscale(:rainbow), colorbar=PlotlyJS.attr(; thickness=20, len=0.5, width=100), line_width=0, line_color="black"))
	geo = PlotlyJS.attr(;
		scope=scope,
		projection_type=projection_type,
		showland=showland,
		landcolor="rgb(217, 217, 217)",
		subunitwidth=1,
		countrywidth=1,
		subunitcolor="rgb(255,255,255)",
		countrycolor="rgb(255,255,255)")
	layout = PlotlyJS.Layout(;
		margin = PlotlyJS.attr(r=0, t=0, b=0, l=0),
		mapbox = PlotlyJS.attr(accesstoken=mapbox_token, style="mapbox://styles/mapbox/satellite-streets-v12"),
		title = title,
		showlegend=false,
		geo=geo)
	p = PlotlyJS.plot(trace, layout)
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	return p
end

# Plot a scatter geo map (categorical)
function plotmap(lon::AbstractVector{T1}, lat::AbstractVector{T1}, color::AbstractVector{T2}; figuredir::AbstractString=".", filename::AbstractString="", format::AbstractString=splitext(filename)[end][2:end], title::AbstractString="", text::AbstractVector=repeat([""], length(lon)), scope::AbstractString="usa", projection_type::AbstractString="albers usa", size::Number=5, showland::Bool=true, kw...) where {T1 <: AbstractFloat, T2 <: Union{Integer,AbstractString,AbstractChar}}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	traces = []
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
			marker=PlotlyJS.attr(; size=size, color=NMFk.colors[k]))
		push!(traces, trace)
	end
	geo = PlotlyJS.attr(
		scope=scope,
		projection_type=projection_type,
		showland=showland,
		landcolor="rgb(217, 217, 217)",
		subunitwidth=1,
		countrywidth=1,
		subunitcolor="rgb(255,255,255)",
		countrycolor="rgb(255,255,255)")
	layout = PlotlyJS.Layout(; title=title, geo=geo)
	p = PlotlyJS.plot(convert(Vector{typeof(traces[1])}, traces), layout)
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	return p
end

function mapbox(df::DataFrames.DataFrame; column::Union{Symbol,AbstractString}="", filename::AbstractString="", title_length::Number=0, kw...)
	regex_lon = r"^[Xx]$|^[Ll]on" # regex for longitude
	regex_lat = r"^[Yy]$|^[Ll]at" # regex for latitude
	rlon = occursin.(regex_lon, names(df))
	rlat = occursin.(regex_lat, names(df))
	if any(rlon) && any(rlat)
		lon = df[!, first(names(df)[rlon])]
		lat = df[!, first(names(df)[rlat])]
	else
		@error("No longitude or latitude column found in the dataframe!")
		return nothing
	end
	if column == ""
		fileroot, fileext = splitext(filename)
		for a in names(df)
			if !(occursin(regex_lon, a) || occursin(regex_lat, a))
				println("Ploting $a ...")
				if filename != ""
					aa = replace(string(a), '/' => Char(0x2215))
					f = fileroot * "_" * aa * fileext
				else
					f = ""
				end
				p = mapbox(lon, lat, df[!, a]; filename=f, title=plotly_title_length(a, title_length), kw...)
				display(p)
			end
		end
	else
		p = mapbox(lon, lat, df[!, column]; filename=filename, title=column, kw...)
		display(p)
	end
end

function mapbox(lon::AbstractVector{T1}, lat::AbstractVector{T1}, M::AbstractMatrix{T2}, names::AbstractVector=["Column $i" for i = eachcol(M)]; filename::AbstractString="", title_length::Number=0, kw...) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	fileroot, fileext = splitext(filename)
	for i in eachindex(names)
		println("Ploting $(names[i]) ...")
		if filename != ""
			aa = replace(string(names[i]), '/' => '\u2215')
			f = fileroot * "_" * aa * fileext
		else
			f = ""
		end
		p = mapbox(lon, lat, M[:,i]; filename=f, title=plotly_title_length(string(names[i]), title_length), kw...)
		display(p)
	end
end

function mapbox(lon::AbstractVector{T1}, lat::AbstractVector{T1}, color::AbstractVector{T2}; title::AbstractString="", text::AbstractVector=repeat([""], length(lon)), dot_size::Number=3,  dot_size_fig::Number=dot_size, lonc::AbstractFloat=minimum(lon)+(maximum(lon)-minimum(lon))/2, font_size::Number=14, font_size_fig::Number=font_size, font_color::AbstractString="black", font_color_fig::AbstractString=font_color, line_color::AbstractString="black", latc::AbstractFloat=minimum(lat)+(maximum(lat)-minimum(lat))/2, zoom::Number=4, zoom_fig::Number=zoom, style="mapbox://styles/mapbox/satellite-streets-v12", mapbox_token=NMFk.mapbox_token, filename::AbstractString="", figuredir::AbstractString=".", format::AbstractString=splitext(filename)[end][2:end], width::Union{Nothing,Int}=nothing, height::Union{Nothing,Int}=nothing, scale::Real=1, legend::Bool=true, colorscale::Symbol=:rainbow, paper_bgcolor::AbstractString="#FFF", showcount::Bool=true, title_length::Number=0) where {T1 <: AbstractFloat, T2 <: AbstractFloat}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	if filename != ""
		if legend
			marker = PlotlyJS.attr(;
				size=dot_size_fig,
				line_width=0,
				line_color=line_color,
				color=color,
				colorscale=NMFk.colorscale(colorscale),
				colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(title, title_length), titlefont=PlotlyJS.attr(size=font_size_fig, color=font_color_fig), tickfont=PlotlyJS.attr(size=font_size_fig, color=font_color_fig))
			)
		else
			marker = PlotlyJS.attr(; size=dot_size_fig, color=color)
		end
		plot = PlotlyJS.scattermapbox(
			lon=lon,
			lat=lat,
			text=text,
			mode="markers",
			hoverinfo="text",
			marker=marker,
			attributionControl=false
		)
		layout = plotly_layout(lonc, latc, zoom_fig; width=width, height=height, title=title, style=style, mapbox_token=mapbox_token)
		p = PlotlyJS.plot(plot, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	if legend
		marker = PlotlyJS.attr(;
			size=dot_size,
			line_width=0,
			line_color=line_color,
			color=color,
			colorscale=NMFk.colorscale(colorscale),
			colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(title, title_length), titlefont=PlotlyJS.attr(size=font_size, color=font_color), tickfont=PlotlyJS.attr(size=font_size, color=font_color))
		)
	else
		marker = PlotlyJS.attr(; size=dot_size, color=color)
	end
	plot = PlotlyJS.scattermapbox(
		lon=lon,
		lat=lat,
		text=text,
		mode="markers",
		hoverinfo="text",
		marker=marker,
		attributionControl=false
	)
	layout = plotly_layout(lonc, latc, zoom; paper_bgcolor=paper_bgcolor, title=title, style=style, mapbox_token=mapbox_token)
	p = PlotlyJS.plot(plot, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
	return p
end

function mapbox(lon::AbstractVector{T1}, lat::AbstractVector{T1}, color::AbstractVector{T2}; title::AbstractString="", text::AbstractVector=repeat([""], length(lon)), dot_size::Number=3,  dot_size_fig::Number=dot_size, lonc::AbstractFloat=minimum(lon)+(maximum(lon)-minimum(lon))/2, font_size::Number=14, font_size_fig::Number=font_size, font_color::AbstractString="black", font_color_fig::AbstractString=font_color, line_color::AbstractString="black", latc::AbstractFloat=minimum(lat)+(maximum(lat)-minimum(lat))/2, zoom::Number=4, zoom_fig::Number=zoom, style="mapbox://styles/mapbox/satellite-streets-v12", mapbox_token=NMFk.mapbox_token, filename::AbstractString="", figuredir::AbstractString=".", format::AbstractString=splitext(filename)[end][2:end], width::Union{Nothing,Int}=nothing, height::Union{Nothing,Int}=nothing, scale::Real=1, legend::Bool=true, colorscale::Symbol=:rainbow, paper_bgcolor::AbstractString="white", showcount::Bool=true, title_length::Number=0) where {T1 <: AbstractFloat, T2 <: Union{Number,Symbol,AbstractString,AbstractChar}}
	@assert length(lon) == length(lat)
	@assert length(lon) == length(color)
	@assert length(lon) == length(text)
	if filename != ""
		traces = []
		for (j, i) in enumerate(unique(sort(color)))
			iz = color .== i
			jj = j % length(NMFk.colors)
			k = jj == 0 ? length(NMFk.colors) : jj
			marker = PlotlyJS.attr(; size=dot_size_fig, color=NMFk.colors[k], colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(repeat("&nbsp;", title_length) * " colorbar " * title, title_length), titlefont=PlotlyJS.attr(size=font_size_fig, color=paper_bgcolor), tickfont=PlotlyJS.attr(size=font_size_fig, color=paper_bgcolor)))
			name = showcount ? "$(string(i)) [$(sum(iz))]" : "$(string(i))"
			trace = PlotlyJS.scattermapbox(;
				lon=lon[iz],
				lat=lat[iz],
				text=text[iz],
				hoverinfo="text",
				name=name,
				marker=marker,
				line_color=line_color,
				showlegend=legend,
				attributionControl=false)
			push!(traces, trace)
		end
		traces = convert(Vector{typeof(traces[1])}, traces)
		layout = plotly_layout(lonc, latc, zoom_fig; paper_bgcolor=paper_bgcolor, font_size=font_size_fig, font_color=font_color_fig, title=title, style=style, mapbox_token=mapbox_token)
		p = PlotlyJS.plot(traces, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
		fn = joinpathcheck(figuredir, filename)
		PlotlyJS.savefig(p, fn; format=format, width=width, height=height, scale=scale)
	end
	traces = []
	for (j, i) in enumerate(unique(sort(color)))
		iz = color .== i
		jj = j % length(NMFk.colors)
		k = jj == 0 ? length(NMFk.colors) : jj
		marker = PlotlyJS.attr(; size=dot_size, color=NMFk.colors[k], colorbar=PlotlyJS.attr(; thicknessmode="pixels", thickness=30, len=0.5, title=plotly_title_length(repeat("&nbsp;", title_length) * " colorbar " * title, title_length), titlefont=PlotlyJS.attr(size=font_size, color=paper_bgcolor), tickfont=PlotlyJS.attr(size=font_size, color=paper_bgcolor)))
		name = showcount ? "$(string(i)) [$(sum(iz))]" : "$(string(i))"
		trace = PlotlyJS.scattermapbox(;
			lon=lon[iz],
			lat=lat[iz],
			text=text[iz],
			hoverinfo="text",
			name=name,
			marker=marker,
			showlegend=true,
			attributionControl=false)
		push!(traces, trace)
	end
	traces = convert(Vector{typeof(traces[1])}, traces)
	layout = plotly_layout(lonc, latc, zoom; paper_bgcolor=paper_bgcolor, title=title, font_size=font_size, font_color=font_color, style=style, mapbox_token=mapbox_token)
	p = PlotlyJS.plot(traces, layout; config=PlotlyJS.PlotConfig(; scrollZoom=true, staticPlot=false, displayModeBar=false, responsive=true))
	return p
end

function mapbox(lon::AbstractFloat=-105.9378, lat::AbstractFloat=35.6870; color::AbstractString="purple", text::AbstractString="EnviTrace LLC", dot_size::Number=12, kw...)
	mapbox([lon], [lat], [color]; text=[text], dot_size=dot_size, legend=false, kw...)
end

# Plotly map layout
function plotly_layout(lonc::Number=-105.9378, latc::Number=35.6870, zoom::Number=4; paper_bgcolor::AbstractString="white", width::Union{Nothing,Int}=nothing, height::Union{Nothing,Int}=nothing, title::AbstractString="", font_size::Number=14, font_color="black", style="mapbox://styles/mapbox/satellite-streets-v12", mapbox_token=NMFk.mapbox_token)
	layout = PlotlyJS.Layout(
		margin = PlotlyJS.attr(r=0, t=0, b=0, l=0),
		title = title,
		autosize = true,
		width = width,
		height = height,
		legend = PlotlyJS.attr(; title_text=title, title_font_size=font_size, itemsizing="constant", font=PlotlyJS.attr(size=font_size, color=font_color), bgcolor=paper_bgcolor),
		paper_bgcolor = paper_bgcolor,
		mapbox = PlotlyJS.attr(
			accesstoken = mapbox_token,
			style = style,
			center = PlotlyJS.attr(lon=lonc, lat=latc),
			zoom = zoom
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