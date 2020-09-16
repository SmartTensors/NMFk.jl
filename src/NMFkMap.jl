import VegaLite
import VegaDatasets
import DataFrames
import Mads

function plotmap(W::AbstractMatrix, H::AbstractMatrix, fips::AbstractVector, dim::Integer=1; casefilename::String="", figuredir::String=".", moviedir::String=".", dates=nothing, plotseries::Bool=true, plotseriesonly::Bool=false, plotpeaks::Bool=!plotseriesonly, plottransients::Bool=!plotseriesonly, movie::Bool=false, name::String="Wave peak", kw...)
	@assert size(W, 2) == size(H, 1)
	Wa, _, _ = NMFk.normalizematrix_col!(W)
	Ha, _, _ = NMFk.normalizematrix_row!(H)
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
	so, si = NMFk.signalorder(Ma, odim)
	nt = dim == 1 ? (Colon(),so) : (so,Colon())
	if dates != nothing
		@assert length(dates) == size(Ma, 1)
		ndates = dates[si]
	else
		ndates = dates
	end
	# signalid = similar(so)
	# for (i, j) in enumerate(so)
	# 	signalid[j] = i
	# end
	plotseries && Mads.plotseries(S[nt...] ./ maximum(S), joinpath(figuredir, casefilename * "-waves.png"); xaxis=dates, names=["$name $(ndates[i])" for i in 1:length(ndates)])
	plotpeaks && NMFk.plotmap(Fa, fips, dim, so; signalid=si, dates=ndates, figuredir=figuredir, casefilename=casefilename, movie=movie, quiet=!movie, kw...)
	if plottransients
		for k in so
			Xe = dim == 1 ? W[:,k:k] * H[k:k,:] : permutedims(W[:,k:k] * H[k:k,:])
			NMFk.plotmap(Xe, fips, 1; dates=dates, figuredir=moviedir, casefilename=casefilename * "-signal-$(k)", datetext="S$(k) ", movie=movie, quiet=!movie, kw...)
		end
	end
end

function plotmap(X::AbstractMatrix, fips::AbstractVector, dim::Integer=1, order::AbstractVector=1:size(X, dim); signalid::AbstractVector=1:size(X, dim), us10m=VegaDatasets.dataset("us-10m"), goodcounties::AbstractVector=trues(length(fips)), dates=nothing, casefilename::String="", figuredir::String=".", title::Bool=false, datetext::String="", titletext::String="", leadingzeros::Integer=1 + convert(Int64, ceil(log10(length(order)))), scheme::String="redyellowgreen", zmin::Number=0, zmax::Number=1, quiet::Bool=false, movie::Bool=false, vspeed::Number=1.0)
	odim = dim == 1 ? 2 : 1
	@assert size(X, odim) == length(fips[goodcounties])
	if dates != nothing
		@assert size(X, dim) == length(dates)
	end
	recursivemkdir(figuredir; filename=false)
	df = DataFrames.DataFrame(FIPS=[fips[goodcounties]; fips[.!goodcounties]])
	for i in order
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(X))
		df[!, :Z] = [vec(X[nt...]); zeros(sum(.!goodcounties))]
		if typeof(signalid[i]) <: Number
			signalidtext = lpad(signalid[i], leadingzeros, '0')
		else
			signalidtext = signalid[i]
		end
		if title || (dates != nothing && titletext != "")
			ttitle = "$(titletext) $(signalidtext)"
			if dates != nothing
				ttitle *= ": $(datetext): $(dates[i])"
			end
			ltitle = ""
		else
			ttitle = nothing
			if dates != nothing
				ltitle =  datetext .* "$(dates[i])"
			else
				ltitle = "$(titletext) $(signalidtext)"
			end
		end
		p = @VegaLite.vlplot(
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
			color={title=ltitle, field="Z", type="quantitative", scale={scheme=scheme, clamp=true, reverse=true, domain=[zmin, zmax]}}
		)
		!quiet && (display(p); println())
		if casefilename != ""
			VegaLite.save(joinpath("$(figuredir)", "$(casefilename)-$(signalidtext).png"), p)
		end
	end
	if casefilename != "" && movie
		makemovie(; moviedir=figuredir, prefix=casefilename, keyword="", numberofdigits=leadingzeros, cleanup=false, vspeed=vspeed)
	end
end

function plotmap(X::AbstractVector, fips::AbstractVector; us10m=VegaDatasets.dataset("us-10m"), goodcounties::AbstractVector=trues(length(fips)), casefilename::String="", figuredir::String=".", title::String="", quiet::Bool=false, scheme::String="category10", zmin::Number=0, zmax::Number=1)
	recursivemkdir(figuredir; filename=false)
	@assert length(X) == length(fips)
	nc = length(unique(sort(X))) + 1
	df = DataFrames.DataFrame(FIPS=[fips[goodcounties]; fips[.!goodcounties]], Z=[X; zeros(sum(.!goodcounties))])
	p = @VegaLite.vlplot(
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
		color={title=title, field="Z", type="ordinal", scale={scheme=vec("#" .*  Colors.hex.(parse.(Colors.Colorant, NMFk.colors), :RGB))[1:nc], reverse=true, domainMax=zmax, domainMin=zmin}}
	)
	!quiet && (display(p); println())
	if casefilename != ""
		VegaLite.save(joinpath("$(figuredir)", "$(casefilename).png"), p)
	end
end