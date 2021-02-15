import Mads
import NMFk
import StatsBase
import Gadfly
import Plotly
import PlotlyJS

function progressive(X::Matrix{T}, windowsize::Int64, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::AbstractString="progressive", load::Bool=true, kw...) where {T}
	checknans = checkarray_nans(X)
	if length(checknans[1]) > 0 || length(checknans[2]) > 0
		@warn("Input matrix contains rows or columns with only NaNs!")
		@show checknans
	end
	@info("NMFk #1: $(casefilename) Window $windowsize")
	W, H, fitquality, robustness, aic = NMFk.execute(X[1:windowsize,:], nkrange, nNMF1; casefilename="$(casefilename)_$(windowsize)", load=load, kw...)
	if windowsize < size(X, 1)
		robustness = Array{T}(undef, 0)
		for k in nkrange
			@info("NMFk #2: $(casefilename) Window $windowsize Signals $k")
			_, _, _, r, _ = NMFk.execute(X, k, nNMF2; Hinit=convert.(T, H[k]), Hfixed=true, casefilename="$(casefilename)_$(windowsize)_all", load=load, kw...)
			push!(robustness, r)
		end
		k = getk(nkrange, robustness)
	else
		k = getk(nkrange, robustness[nkrange])
	end
	return k
end

function progressive(X::Matrix{T}, windowsize::Vector{Int64}, window_k::Vector{Int64}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::AbstractString="progressive", load::Bool=true, kw...) where {T}
	@assert length(windowsize) == length(window_k)
	checknans = checkarray_nans(X)
	if length(checknans[1]) > 0 || length(checknans[2]) > 0
		@warn("Input matrix contains rows or columns with only NaNs!")
		@show checknans
	end
	# @assert all(map(i->sum(.!isnan.(X[i, :])) > 0, 1:size(X, 1)))
	# @assert all(map(i->sum(.!isnan.(X[:, i])) > 0, 1:size(X, 2)))
	# @show map(i->sum(.!isnan.(X[i, :])), 1:size(X, 1))
	# @show map(i->sum(.!isnan.(X[:, i])), 1:size(X, 2))
	for (i, ws) in enumerate(windowsize)
		k = window_k[i]
		@info("NMFk #1: $(casefilename) Window $ws Signals $k")
		W, H, fitquality, robustness, aic = NMFk.execute(X[1:ws,:], k, nNMF1; casefilename="$(casefilename)_$(ws)", load=load, kw...)
		if ws < size(X, 1)
			@info("NMFk #2: $(casefilename) Window $ws Signals $k")
			NMFk.execute(X, k, nNMF2; Hinit=convert.(T, H), Hfixed=true, casefilename="$(casefilename)_$(ws)_all", load=load, kw...)
		end
	end
	return window_k
end

function progressive(X::Matrix{T}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::AbstractString="progressive", load::Bool=true, kw...) where {T}
	checknans = checkarray_nans(X)
	if length(checknans[1]) > 0 || length(checknans[2]) > 0
		@warn("Input matrix contains rows or columns with only NaNs!")
		@show checknans
	end
	# @assert all(map(i->sum(.!isnan.(X[i, :])) > 0, 1:size(X, 1)))
	# @assert all(map(i->sum(.!isnan.(X[:, i])) > 0, 1:size(X, 2)))
	# @show map(i->sum(.!isnan.(X[i, :])), 1:size(X, 1))
	# @show map(i->sum(.!isnan.(X[:, i])), 1:size(X, 2))
	window_k = Array{Int64}(undef, 0)
	for ws in windowsize
		@info("NMFk #1: $(casefilename) Window $ws")
		W, H, fitquality, robustness, aic = NMFk.execute(X[1:ws,:], nkrange, nNMF1; casefilename="$(casefilename)_$(ws)", load=load, kw...)
		k = getk(nkrange, robustness[nkrange])
		push!(window_k, k)
		if ws < size(X, 1)
			@info("NMFk #2: $(casefilename) Window $ws: Best $k")
			NMFk.execute(X, k, nNMF2; Hinit=convert.(T, H[k]), Hfixed=true, casefilename="$(casefilename)_$(ws)_all", load=load, kw...)
		end
	end
	return window_k
end

function progressive(X::Vector{Matrix{T}}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::AbstractString="progressive", load::Bool=true, kw...) where {T}
	window_k = Array{Int64}(undef, 0)
	for ws in windowsize
		@info("NMFk #1: $(casefilename) Window $ws")
		normalizevector = vcat(map(i->fill(NMFk.maximumnan(X[i][1:ws,:]), ws), 1:length(X))...)
		W, H, fitquality, robustness, aic = NMFk.execute(vcat([X[i][1:ws,:] for i = 1:length(X)]...), nkrange, nNMF1; normalizevector=normalizevector,casefilename="$(casefilename)_$(ws)", load=load, kw...)
		k = getk(nkrange, robustness[nkrange])
		push!(window_k, k)
		# global wws = 1
		# global wwe = ws
		# for i = 1:length(X)
		# 	display(X[i][1:ws,:] .- W[k][wws:wwe,:] * H[k])
		# 	wws += ws
		# 	wwe += ws
		# end
		if ws < size(X[1], 1)
			@info("NMFk #2: $(casefilename) Window $ws: Best $k")
			normalizevector = vcat(map(i->fill(NMFk.maximumnan(X[i]), size(X[1], 1)), 1:length(X))...)
			Wa, Ha, fitquality, robustness, aic = NMFk.execute(vcat([X[i] for i = 1:length(X)]...), k, nNMF2; Hinit=convert.(T, H[k]), Hfixed=true, normalizevector=normalizevector, casefilename="$(casefilename)_$(ws)_all", load=load, kw...)
			# global wws = 1
			# global wwe = size(X[1], 1)
			# for i = 1:length(X)
			# 	display((X[i] .- Wa[wws:wwe,:] * Ha)
			# 	wws += size(X[1], 1)
			# 	wwe += size(X[1], 1)
			# end
		end
	end
	return window_k
end

function progressive(syears::AbstractVector, eyears::AbstractVector, df::DataFrames.DataFrame, df_header::DataFrames.DataFrame, api::AbstractVector; nNMF::Integer=100, load::Bool=true, problem::AbstractString="gaswellshor", figuredirdata::AbstractString="figures-data-eagleford", resultdir::AbstractString="results-nmfk-eagleford", figuredirresults::AbstractString="figures-nmfk-eagleford", scale::Bool=false, normalize::Bool=true, plotseries::Bool=false, plotr2pred::Bool=true, kw...)
	@assert length(syears) == length(eyears)

	startdate = minimum(df[!, :ReportDate])
	for (qq, y) in enumerate(syears)
		if typeof(y) <: Dates.Date
			enddate = y
			period = "$(enddate)"
			period_pred = "$(enddate)"
		else
			enddate = Dates.Date(y - 1, 12, 1)
			period = "$(y)"
			period_pred = "$(y)"
		end
		if typeof(eyears[qq]) <: Dates.Date
			enddate_pred = eyears[qq]
			period_pred *= "-$(enddate_pred)"
		else
			enddate =  Dates.Date(eyears[qq] - 1, 12, 1)
			period_pred *= "-$(qq)"
		end
 		dates = collect(startdate:Dates.Month(1):enddate)
		dates_pred = collect(startdate:Dates.Month(1):enddate_pred)

		gas_data, existing_wells = NMFk.df2matrix(df, api, dates, :WellGas)
		nw = sum(existing_wells)
		Mads.plotseries(gas_data, "$(figuredirdata)-$(problem)/data_$(period).png"; xaxis=dates)

		@info "$(period): Number of wells $nw"

		well_x = Array{Float32}(undef, 0)
		well_y = Array{Float32}(undef, 0)
		for w in api[existing_wells]
			iwell = findall((in)(w), df_header[!, :API])
			if iwell !== nothing
				push!(well_x, df_header[!, :Lon][iwell[1]])
				push!(well_y, df_header[!, :Lat][iwell[1]])
			else
				@info("Well $w coordinates are missing!")
			end
		end

		gas_train, startdates_train, enddates_train = NMFk.df2matrix_shifted(df, api[existing_wells], dates, :WellGas)
		@info("Training matrix size: $(size(gas_train))")
		@info("Training start date: $(minimum(startdates_train))")
		@info("Training end   date: $(minimum(startdates_train))")

		Mads.plotseries(gas_train, "$(figuredirdata)-$(problem)/data-shifted-$(period).png"; xmax=size(gas_train, 1))

		gas_pred, startdates_pred, enddates_pred = NMFk.df2matrix_shifted(df, api[existing_wells], dates_pred, :WellGas)
		@info("Prediction matrix size: $(size(gas_pred))")
		@info("Prediction start date: $(minimum(startdates_pred))")
		@info("Prediction end   date: $(minimum(startdates_pred))")

		@assert startdates_train == startdates_pred

		Mads.plotseries(gas_pred, "$(figuredirdata)-$(problem)/data-shifted-$(period)-all.png"; xmax=size(gas_pred, 1))

		train_window = size(gas_train, 1)
		pred_window = size(gas_pred, 1) - size(gas_train, 1)

		ds = [train_window]
		@info("Training window: $train_window")
		@info("Prediction window: $(pred_window)")
		dk = NMFk.progressive(gas_train, ds, 2:5, nNMF; resultdir="$(resultdir)-$(problem)", casefilename="gas_$(period)", method=:simple, load=load, scale=scale, normalize=normalize, kw...)

		dk = [i for i=2:dk[1]]
		ds = repeat(ds, length(dk))

		@info("Optimal number of signals: $dk Training window sizes: $ds")

		Mads.mkdir("$(figuredirresults)-$(problem)")
		for j = 1:length(ds)
			if ds[j] != train_window
				Wall, Hall, fitquality, robustness, aic = NMFk.load(dk[j], nNMF; resultdir="$(resultdir)-$(problem)", casefilename="gas_$(period)_$(ds[j])_all")
			else
				Wall, Hall, fitquality, robustness, aic = NMFk.load(dk[j], nNMF; resultdir="$(resultdir)-$(problem)", casefilename="gas_$(period)_$(ds[j])")
			end

			Mads.plotseries(Wall, "$(figuredirresults)-$(problem)/data-signals-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred).png"; title="", ymin=0.0)
			Oall = Wall * Hall
			if sum(Oall) == 0
				@warn("Something is wrong!")
				continue
			end
			l = size(Oall, 1)
			hovertext = Vector{String}(undef, 0)
			rate_t = Vector{Float64}(undef, 0)
			gas_t = Vector{Float64}(undef, 0)
			gas_p = Vector{Float64}(undef, 0)
			rate_ta = Vector{Float64}(undef, 0)
			gas_ta = Vector{Float64}(undef, 0)
			gas_pa = Vector{Float64}(undef, 0)
			for (i, s) in enumerate(api[existing_wells])
				q = findlast(.!isnan.(gas_pred[:,i]))
				p = (q > l) ? l : q
				op = gas_pred[1:p,i]
				ip = .!isnan.(op)
				if sum(ip) == 0
					continue
				end
				truth = sum(op[ip])
				pred = sum(Oall[1:p,i][ip])
				push!(rate_ta, truth / sum(ip))
				push!(gas_ta, truth)
				push!(gas_pa, pred)
				push!(hovertext, "Well: $(s)<br>Start Date: $(startdates_train[i])<br>Total Gas: $(round(truth; sigdigits=3))<br>Predicted: $(round(pred; sigdigits=3))")
				if startdates_train[i] <= enddate && enddates_pred[i] > enddate
					r = length(startdates_train[i]:Dates.Month(1):enddate)
					op = gas_pred[r:p,i]
					ip = .!isnan.(op)
					truth = sum(op[ip])
					pred = sum(Oall[r:p,i][ip])
					push!(rate_t, truth / sum(ip))
					push!(gas_t, truth)
					push!(gas_p, pred)
					perror = abs(pred - truth) / truth * 100
					if mod(i, 10) == 1
						ending =  abs(perror) > 50 ? "-bad" : ""
						# ending = ""
						gm = [Gadfly.layer(xintercept=[enddate], Gadfly.Geom.vline(color=["darkgray"], size=[4Gadfly.pt]))]
						Mads.plotseries([Oall[1:p,i] gas_pred[1:p,i]], "$(figuredirresults)-$(problem)$(ending)/data-prediction-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred)-well-$s.png"; title="Well $(s) : $(period_pred)",  names=["Prediction $(round(pred; sigdigits=3))", "Truth $(round(truth; sigdigits=3))"], colors=["blue", "red"], ymin=0.0, xmin=startdates_train[i], xmax=startdates_train[i] + Dates.Month(p-1), xaxis=collect(startdates_train[i]:Dates.Month(1):startdates_train[i] + Dates.Month(p-1)), gm=gm, quiet=!plotseries, dpi=100)
					end
				end
			end
			r2 = NMFk.r2(gas_t, gas_p)
			r2a = NMFk.r2(gas_ta, gas_pa)
			@info("Window $period: Training size $(ds[j]) Truth size: $(length(gas_t)) Prediction size: $(length(gas_p)) R2 (pred): $r2 R2 (all) $r2a")

			if load && isfile("$(resultdir)-$(problem)/gas_$(period)_$(ds[j])-$(dk[j])-$(nNMF)-assignments.jld2")
				c_gas = FileIO.load("$(resultdir)-$(problem)/gas_$(period)_$(ds[j])-$(dk[j])-$(nNMF)-assignments.jld2", "c_gas")
			else
				c_gas = NMFk.labelassignements(NMFk.robustkmeans(Hall, dk[j])[1].assignments)
				FileIO.save("$(resultdir)-$(problem)/gas_$(period)_$(ds[j])-$(dk[j])-$(nNMF)-assignments.jld2", "c_gas", c_gas)
			end

			p = PlotlyJS.plot(NMFk.plot_wells(well_x, well_y, c_gas; hover=hovertext), Plotly.Layout(title="Gas $(period): $(dk[j]) types"))
			PlotlyJS.savehtml(p, "$(figuredirresults)-$(problem)/data-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred).html", :remote)

			for ct in sort(unique(c_gas))
				i = c_gas .== ct
				@info "Type $(ct) wells: $(sum(i))"
				Mads.plotseries(gas_train[:, i], "$(figuredirresults)-$(problem)/data-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred)_type_$(ct).png")
			end

			for ct in sort(unique(c_gas))
				i = c_gas .== ct
				@info "Type $(ct) wells: $(sum(i))"
				@info "Formation"
				display(NMFk.bincount(df_header[!, :Formation][findall((in)(api[existing_wells][i]), df_header[!, :API])]; cutoff=1))
				@info "Operator"
				display(NMFk.bincount(df_header[!, :Operator][findall((in)(api[existing_wells][i]), df_header[!, :API])]; cutoff=1))
				@info "Well type"
				display(NMFk.bincount(df_header[!, :Orientation][findall((in)(api[existing_wells][i]), df_header[!, :API])]; cutoff=1))
			end

			NMFk.histogram(log10.(rate_ta), c_gas; title="Gas", xtitle="Gas Monthly Rate", ytitle="Count", filename="$(figuredirresults)-$(problem)/data-histogram-rate-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred).png", proportion=false, joined=false, separate=true, xlabelmap=i->"10<sup>$i</sup>", refine=2)

			sd = map(i->length(startdate:Dates.Month(1):i), startdates_train)
			NMFk.histogram(sd, c_gas; title="Start Date", xtitle="", ytitle="Count", filename="$(figuredirresults)-$(problem)/data-histogram-startdate-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred).png", proportion=false, joined=false, separate=true, xmap=i->Dates.epochms2datetime(Dates.datetime2epochms(Dates.DateTime(startdate + Dates.Month(floor(i))))), refine=2)

			if length(gas_p) > 0
				NMFk.plotscatter(gas_ta, gas_pa; filename="$(figuredirresults)-$(problem)/data-scatter-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred)_all.png", title="Gas $(period_pred): Window $(ds[j]) months r2=$(round(r2a; sigdigits=3)) count=$(length(gas_ta))", xtitle="Truth", ytitle="Prediction", line=true)
				plotr2pred && NMFk.plotscatter(gas_t, gas_p; filename="$(figuredirresults)-$(problem)/data-scatter-$(ds[j])-$(dk[j])-$(nNMF)-$(period_pred).png", title="Gas $(period_pred): Window $(ds[j]) months r2=$(round(r2; sigdigits=3)) count=$(length(gas_t))", xtitle="Truth", ytitle="Prediction", line=true)
			else
				@warn("No data!")
				@warn("Something went wrong")
			end
		end
	end
end

function getk(nkrange::Union{AbstractRange{T1},AbstractVector{T1}}, robustness::AbstractVector{T2}, cutoff::Number=0.25) where {T1 <: Integer, T2 <: Number}
	@assert length(nkrange) == length(robustness)
	if all(isnan.(robustness))
		return 0
	end
	if length(nkrange) == 1
		k = nkrange[1]
	else
		kn = findlast(i->i > cutoff, robustness)
		if kn === nothing
			inan = isnan.(robustness)
			robustness[inan] .= -Inf
			kn = findmax(robustness)[2]
			robustness[inan] .= NaN
		end
		k = nkrange[kn]
	end
	return k
end

function getks(nkrange::Union{AbstractRange{T1},AbstractVector{T1}}, robustness::AbstractVector{T2}, cutoff::Number=0.25) where {T1 <: Integer, T2 <: Number}
	@assert length(nkrange) == length(robustness)
	if all(isnan.(robustness))
		return []
	end
	if length(nkrange) == 1
		k = [nkrange[1]]
	else
		kn = findall(i->i > cutoff, robustness)
		if (length(kn) == 0)
			inan = isnan.(robustness)
			robustness[inan] .= -Inf
			k = nkrange[findmax(robustness)[2]]
			robustness[inan] .= NaN
		else
			k = nkrange[kn]
		end
	end
	return k
end

function getks(nkrange::Union{AbstractRange{T1},AbstractVector{T1}}, F::AbstractVector{T2}, map=Colon(), cutoff::Number=0.25) where {T1 <: Integer, T2 <: AbstractArray}
	@assert length(nkrange) == length(F)
	if all(isnan.(robustness))
		return []
	end
	if length(nkrange) == 1
		kn = [nkrange[1]]
	else
		kn = Vector{Int64}(undef, 0)
		for (i, k) in enumerate(nkrange)
			if size(F[i], 1) == k
				M = F[i] ./ maximum(F[i]; dims=2)
				any(M[:,map] .> cutoff) && push!(kn, k)
			elseif size(F[i], 2) == k
				M = F[i] ./ maximum(F[i]; dims=1)
				any(M[map,:] .> cutoff) && push!(kn, k)
			end
		end
	end
	return kn
end