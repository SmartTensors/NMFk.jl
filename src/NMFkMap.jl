import VegaLite
import VegaDatasets
import DataFrames

function plotmap(W::AbstractMatrix, H::AbstractMatrix, fips::AbstractVector, dim::Integer=1; dates=nothing, kw...)
	Wa, _, _ = NMFk.normalizematrix_col!(W)
	Ha, _, _ = NMFk.normalizematrix_row!(H)
	if dim == 1
		odim = 2
		so, si = NMFk.signalorder(Wa, odim)
		if dates != nothing
			dates=dates[si]
		end
		NMFk.plotmap(Ha, fips, dim, so; dates=dates, kw...)
	else
		odim = 1
		so, si = NMFk.signalorder(Ha, odim)
		if dates != nothing
			dates=dates[si]
		end
		NMFk.plotmap(Wa, fips, dim, so; dates=dates, kw...)
	end
end

function plotmap(X::AbstractMatrix, fips::AbstractVector, dim::Integer=1, order=1:size(X, dim); us10m=VegaDatasets.dataset("us-10m"), goodcounties=trues(length(fips)), dates=nothing, casefilename="", figuredir=".", title::Bool=false, datetext="Date", titletext="", leadingzeros=2, quiet::Bool=false, scheme="redyellowgreen", zmin=0, zmax=1)
	recursivemkdir(figuredir; filename=false)
	for (k, i) in enumerate(order)
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(X))
		df = DataFrames.DataFrame(FIPS=[fips[goodcounties]; fips[.!goodcounties]], Z=[vec(X[nt...]); zeros(sum(.!goodcounties))])
		if title || (dates != nothing && titletext != "")
			ttitle = "$(titletext) $(lpad(k, leadingzeros, '0'))"
			if dates != nothing
				ttitle *= ": $(datetext): $(dates[i])"
			end
			ltitle = ""
		else
			ttitle = nothing
			if dates != nothing
				ltitle = "$(dates[i])"
			else
				ltitle = "$(titletext) $(lpad(k, leadingzeros, '0'))"
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
			color={title=ltitle, field="Z", type="quantitative", scale={scheme=scheme, reverse=true, domainMax=zmax, domainMin=zmin}}
		)
		!quiet && (display(p); println())
		if casefilename != ""
			VegaLite.save(joinpath("$(figuredir)", "$(casefilename)-$(lpad(k, 2, '0')).png"), p)
		end
	end
end