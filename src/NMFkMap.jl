import VegaLite
import VegaDatasets
import DataFrames
import Mads
import PlotlyJS

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
			c = Mads.plotseries(S[nt...] ./ maximum(S); xaxis=dates, names=["S$i $(ndates[k])" for (i,k) in enumerate(signalorderassignments)], code=true, quiet=true)
			progressbar = NMFk.make_progressbar_2d(c)
			for i = eachindex(dates)
				p = progressbar(i, true, 1, dates[1])
				Gadfly.draw(Gadfly.PNG(joinpathcheck(moviedir, casefilename * "-progressbar-$(lpad(i, 6, '0')).png"), hsize, vsize, dpi=dpi), p)
				!quiet && (@info dates[i]; Mads.display(p; gw=hsize, gh=vsize))
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

function plotmap(x::AbstractVector{T}, y::AbstractVector{T}, c::AbstractVector{T}; figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", size=5, text=repeat([""], length(x))) where T <: Real
	@assert length(x) == length(y)
	@assert length(x) == length(text)
	trace = PlotlyJS.scattergeo(; locationmode="USA-states",
		lon=x,
		lat=y,
		hoverinfo="text",
		text=text,
		marker=Plotly.attr(; size=size, color=c, colorscale=NMFk.colorscale(:rainbow), colorbar=Plotly.attr(; thickness=20, width=100), line_width=0, line_color="black"))
	geo = PlotlyJS.attr(scope="usa",
		projection_type="albers usa",
		showland=true,
		landcolor="rgb(217, 217, 217)",
		subunitwidth=1,
		countrywidth=1,
		subunitcolor="rgb(255,255,255)",
		countrycolor="rgb(255,255,255)")
	layout = PlotlyJS.Layout(; title=title, showlegend=false, geo=geo)
	p = PlotlyJS.plot(trace, layout)
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		recursivemkdir(fn)
		PlotlyJS.savefig(p, fn; format="html")
	end
	return p
end

function plotmap(x::AbstractVector{T1}, y::AbstractVector{T1}, c::AbstractVector{T2}; figuredir::AbstractString=".", filename::AbstractString="", title::AbstractString="", size=5, text=repeat([""], length(x))) where {T1 <: Real, T2 <: Union{Integer,AbstractString,AbstractChar}}
	@assert length(x) == length(y)
	@assert length(x) == length(text)
	traces = []
	for (j, i) in enumerate(unique(sort(c)))
		iz = c .== i
		jj = j % length(NMFk.colors)
		k = jj == 0 ? length(NMFk.colors) : jj
		trace = PlotlyJS.scattergeo(; locationmode="USA-states",
			lon=x[iz],
			lat=y[iz],
			hoverinfo="text",
			text=text[iz],
			name="$i $(sum(iz))",
			marker=Plotly.attr(; size=size, color=NMFk.colors[k]))
		push!(traces, trace)
	end
	geo = PlotlyJS.attr(scope="usa",
		projection_type="albers usa",
		showland=true,
		landcolor="rgb(217, 217, 217)",
		subunitwidth=1,
		countrywidth=1,
		subunitcolor="rgb(255,255,255)",
		countrycolor="rgb(255,255,255)")
	layout = PlotlyJS.Layout(; title=title, geo=geo)
	p = PlotlyJS.plot(convert(Array{typeof(traces[1])}, traces), layout)
	if filename != ""
		fn = joinpathcheck(figuredir, filename)
		recursivemkdir(fn)
		PlotlyJS.savefig(p, fn; format="html")
	end
	return p
end