import VegaLite
import VegaDatasets
import DataFrames
import Mads
import PlotlyJS

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
			println("Plotting $(varname) ...")
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
			marker=PlotlyJS.attr(; size=marker_size, color=color, colorscale=NMFk.colorscale(:turbo), colorbar=PlotlyJS.attr(; thickness=20, len=0.5, width=100, tickfont_size=font_size), line_width=0, line_color="black"))
		return trace
	end
	if filename != ""
		p_fig = PlotlyJS.plot(trace_fig(marker_size_fig, font_size_fig), layout_fig(font_size_fig))
		fn = joinpathcheck(figuredir, filename)
		safe_savefig(p_fig, fn; format=format, width=width, height=height, scale=scale)
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
		safe_savefig(p, fn; format=format, width=width, height=height, scale=scale)
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