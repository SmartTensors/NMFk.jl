import Statistics
import StatsBase
import DataFrames
import Missings

function checkarray(X::AbstractArray{T,N}, cutoff::Integer=0; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func) where {T <: Number, N}
	rangeentry = Vector{AbstractUnitRange{Int64}}(undef, N)
	# min_firstentry = Vector{Int64}(undef, N)
	# max_lastentry = Vector{Int64}(undef, N)
	max_record_length = Vector{Int64}(undef, N)
	for d = 1:N
		@info("Dimension $(d) ...")
		dd = size(X, d)
		println("Dimension $(d): size: $(dd)")
		firstentrys = Vector{Int64}(undef, dd)
		lastentrys = Vector{Int64}(undef, dd)
		record_length = Vector{Int64}(undef, dd)
		bad_indices = Vector{Int64}(undef, 0)
		for i = 1:dd
			nt = ntuple(k->(k == d ? i : Colon()), N)
			ix = X[nt...]
			if i == 1
				println("Dimension $(d): slice: $(size(ix))")
			end
			firstentry = Base.findfirst(funcfirst.(ix))
			# @show nt
			# @show ix
			# @show firstentry
			if !isnothing(firstentry)
				firstentrys[i] = firstentry[1]
				lastentry = Base.findlast(funclast.(ix[firstentrys[i]:end]))
				if !isnothing(lastentry)
					lastentrys[i] = lastentry[1] + firstentrys[i] - 1
					record_length[i] = lastentry
				else
					lastentrys[i] = length(ix)
					record_length[i] = length(ix[firstentry:end])
				end
			else
				firstentrys[i] = length(ix)
				lastentrys[i] = 0
				record_length[i] = 0
			end
			if record_length[i] <= cutoff
				push!(bad_indices, i)
			end
		end
		if length(bad_indices) > 0
			println("Dimension $(d): Bad indices: $bad_indices")
		else
			println("Dimension $(d): No bad indices!")
		end
		ir = sortperm(record_length)
		if length(record_length[ir]) > 50
			println("Dimension $(d): Worst 15 entry counts: $(record_length[ir][1:15])")
			println("Dimension $(d): Best 15 entry counts: $(record_length[ir][end-15:end])")
		else
			println("Dimension $(d): Entry counts: $(record_length)")
		end
		mfe = minimum(firstentrys)
		mle = maximum(lastentrys)
		mrl = maximum(record_length)
		println("Dimension $(d): Maximum entry counts: $(mrl)")
		println("Dimension $(d): Minimum first  entry: $(mfe)")
		println("Dimension $(d): Maximum last   entry: $(mle)")
		# @info("Dimension $(d): First entry: $(firstentrys)")
		# @info("Dimension $(d): Last  entry: $(lastentrys)")
		# md[d] = length(bad_indices) > 0 ? bad_indices[1] : 0
		max_record_length[d] = mrl
		rangeentry[d] = mfe:mle
	end
	return rangeentry
end

checkarray_nans(X::AbstractArray; kw...) = checkarrayentries(X; kw...)
checkarray_zeros(X::AbstractArray; kw...) = checkarrayentries(X, i->i>0; kw...)
checkarray_count(X::AbstractArray; kw...) = checkarrayentries(X; ecount=true, kw...)

function checkarrayentries(X::AbstractArray{T,N}, func::Function=.!isnan; quiet::Bool=false, debug::Bool=false, good::Bool=false, ecount::Bool=false, cutoff::Integer=0) where {T <: Number, N}
	local flag = true
	return_indices = Vector{Vector{Int64}}(undef, N)
	for d = 1:N
		!quiet && @info("Dimension $(d) ...")
		selected_indices = Vector{Int64}(undef, 0)
		if ecount
			acount = Vector{Int64}(undef, 0)
		end
		for i in axes(X, d)
			nt = ntuple(k->(k == d ? i : Colon()), N)
			c = sum(func.(X[nt...]))
			if ecount
				push!(acount, c)
			end
			flagi = c > cutoff
			if good
				flagi && push!(selected_indices, i)
			else
				!flagi && push!(selected_indices, i)
			end
			flag = flag && flagi
		end
		if ecount
			return_indices[d] = acount
			st = "count"
		else
			return_indices[d] = selected_indices
			st = good ? "good indices" : "bad indices"
		end
		if !quiet
			if length(return_indices[d]) > 0
				println("Dimension $(d) $(st): $(return_indices[d])")
				if debug
					nt = ntuple(k->(k == d ? return_indices[d] : Colon()), N)
					display(nt)
					display(X[nt...])
				end
			else
				println("Dimension $(d): No $(st):")
			end
		end
	end
	return return_indices
end

checkcols(x::AbstractMatrix; kw...) = checkmatrix(x::AbstractMatrix, 2; kw...)
checkrows(x::AbstractMatrix; kw...) = checkmatrix(x::AbstractMatrix, 1; kw...)

function maskvector(x::AbstractVector)
	ism = .!ismissing.(x) .& .!isnothing.(x)
	xt = identity.(x[ism])
	if eltype(xt) <: AbstractFloat
		isn = .!isnan.(xt)
		xt = xt[isn]
		ism[ism .== 1] .= isn
	end
	return ism
end

function checkmatrix(df::DataFrames.DataFrame; names=names(df), kw...)
	@assert length(names) == DataFrames.ncol(df)
	return checkmatrix(Matrix(df), 2; names=names, kw...)
end

function checkmatrix(x::AbstractMatrix, dim=2; quiet::Bool=false, correlation_test::Bool=true, correlation_cutoff::Number=0.99, norm_cutoff::Number=0.01, skewness_cutoff::Number=1., name::AbstractString=dim == 2 ? "Column" : "Row", names::AbstractVector=["$name $i" for i=axes(x, dim)], masks::Bool=true)
	names = String.(names)
	mlength = maximum(length.(names))
	na = size(x, dim)
	ilog = Vector{Int64}(undef, 0)
	icor = Vector{Int64}(undef, 0)
	isame = Vector{Int64}(undef, 0)
	inans = Vector{Int64}(undef, 0)
	izeros = Vector{Int64}(undef, 0)
	ineg = Vector{Int64}(undef, 0)
	istatic = Vector{Int64}(undef, 0)
	istring = Vector{Int64}(undef, 0)
	idates = Vector{Int64}(undef, 0)
	iany = Vector{Int64}(undef, 0)
	for i = axes(x, dim)
		!quiet && print("$(Base.text_colors[:cyan])$(Base.text_colors[:bold])$(NMFk.sprintf("%-$(mlength)s", names[i])):$(Base.text_colors[:normal]) ")
		nt = ntuple(k->(k == dim ? i : Colon()), 2)
		vo = x[nt...]
		isvalue = maskvector(vo)
		if sum(isvalue) == 0
			!quiet && println("$(Base.text_colors[:magenta])$(Base.text_colors[:bold])Only missing values (NaNs)$(Base.text_colors[:normal])")
			push!(inans, i)
			continue
		end
		v = identity.(vo[isvalue])
		if eltype(v) <: Number
			vmin = minimum(v)
			vmax = maximum(v)
			skew = StatsBase.skewness(v)
			luv = length(unique(v))
			print("min: $(Printf.@sprintf("%12.7g", vmin)) max: $(Printf.@sprintf("%12.7g", vmax)) skewness: $(Printf.@sprintf("%12.7g", skew)) count: $(Printf.@sprintf("%12d", length(v))) unique: $(Printf.@sprintf("%12d", luv))")
			skip_corr_test = false
			if sum(v) == 0
				!quiet && print(" <- only zeros!")
				skip_corr_test = true
				push!(izeros, i)
			elseif any(v .< 0)
				!quiet && print(" <- negative values!")
				push!(ineg, i)
			end
			if vmin ≈ vmax
				!quiet && print(" <- constant!")
				skip_corr_test = true
				push!(istatic, i)
			elseif length(unique(v)) == 2
				!quiet && print(" <- boolean?!")
			elseif abs(skew) > skewness_cutoff && length(unique(v)) > 2
				!quiet && print(" <- very skewed; log-transformation recommended!")
				push!(ilog, i)
			end
			println()
			if !skip_corr_test && correlation_test
				for j = 1:na
					if i == j
						continue
					end
					nt2 = ntuple(k->(k == dim ? j : Colon()), 2)
					vo2 = x[nt2...]
					isvalue2 = maskvector(vo2)
					if sum(isvalue2) == 0 # only missing values
						continue
					end
					v2 = identity.(vo2[isvalue2])
					if !(eltype(v2) <: Number)
						continue
					end
					if isvalue !== isvalue2
						isvalue_all = isvalue .& isvalue2
						if sum(isvalue_all) == 0
							continue
						end
						v = vo[isvalue_all]
						v2 = vo2[isvalue_all]
						comparison_ratio = sum(isvalue_all) / sum(isvalue)
						comparison_size = "$(sum(isvalue_all)) out of $(sum(isvalue))"
					else
						comparison_ratio = 1
						comparison_size = "$(sum(isvalue)) out of $(sum(isvalue))"
					end
					if isvalue == isvalue2 && v == v2
						!quiet && println("- equivalent with $(Base.text_colors[:cyan])$(Base.text_colors[:bold])$(names[j])$(Base.text_colors[:normal]) (comparison size = $(comparison_size))!")
						if i > j
							push!(isame, j)
						end
					elseif sum(isvalue_all) > 2 && comparison_ratio > 0.5 && Statistics.norm(v .- v2) < norm_cutoff || all(v .≈ v2)
						!quiet && println("- similar with $(Base.text_colors[:cyan])$(Base.text_colors[:bold])$(names[j])$(Base.text_colors[:normal]) (comparison size = $(comparison_size))!")
						if i > j
							push!(icor, j)
						end
					elseif sum(isvalue_all) > 3 && comparison_ratio > 0.5 && (correlation = abs(Statistics.cor(v, v2))) > correlation_cutoff
						correlation = round(correlation, digits=4)
						!quiet && println("- correlated with $(Base.text_colors[:cyan])$(Base.text_colors[:bold])$(names[j])$(Base.text_colors[:normal]) (correlation = $(correlation)) (comparison size = $(comparison_size))!")
						if i > j
							push!(icor, j)
						end
					end
				end
			end
		elseif eltype(v) <: Dates.Date
			vmin = minimum(v)
			vmax = maximum(v)
			luv = length(unique(v))
			println("min: $(Printf.@sprintf("%12s", vmin)) max: $(Printf.@sprintf("%12s", vmax)) skewness: $(Printf.@sprintf("%12s", "---")) count: $(Printf.@sprintf("%12d", length(v))) unique: $(Printf.@sprintf("%12d", luv))")
			push!(idates, i)
		elseif eltype(v) <: AbstractString
			push!(istring, i)
			print("$(Base.text_colors[:yellow])$(Base.text_colors[:bold])String:$(Base.text_colors[:normal]) ")
			u = sort(String.(unique(v)))
			if length(u) == 1
				!quiet && println("$(u) <- non-numeric constant!")
				push!(istatic, i)
			else
				up = deepcopy(u)
				for (i, s) in enumerate(u)
					if length(s) > 24
						up[i] = s[1:20] * " ..."
					else
						if length(s) == 0
							up[i] = rpad("<empty>", 24)
						else
							up[i] = rpad(s, 24)
						end
					end
				end
				if !quiet
					println("$(length(u)) unique values:")
					count = [sum(i .== v) for i in u]
					isort = sortperm(count; rev=true)
					if length(u) > 20
						for i in 1:5
							println("$(up[isort][i]): count = $(count[isort][i])")
						end
						println("...")
						for i in eachindex(up)[end-4:end]
							println("$(up[isort][i]): count = $(count[isort][i])")
						end
					else
						for i in eachindex(up)
							println("$(up[isort][i]): count = $(count[isort][i])")
						end
					end
				end
			end
		else
			!quiet && println("$(Base.text_colors[:red])$(Base.text_colors[:bold])Unknown type:$(Base.text_colors[:normal]) $(eltype(v)): $(unique(v))!")
			push!(iany, i)
		end
	end
	icor = unique(sort(icor))
	if !quiet
		@info("Attribute summary:")
		println("Log-transformation recommended: $(length(ilog))")
		println("Correlated with other attributes: $(length(icor))")
		println("Include negative values: $(length(ineg))")
		println("Equivalent with other attributes: $(length(isame))")
		println("Contain only missing values: $(length(inans))")
		println("Contain only zeros: $(length(izeros))")
		println("Constant entries: $(length(istatic))")
		println("String entries: $(length(istring))")
		println("Date entries: $(length(istring))")
		println("Any entries: $(length(iany))")
	end
	if masks
		mlog = falses(na)
		mcor = falses(na)
		msame = falses(na)
		mnans = falses(na)
		mzeros = falses(na)
		mneg = falses(na)
		mstatic = falses(na)
		mstring = falses(na)
		mdates = falses(na)
		many = falses(na)
		mlog[ilog] .= true
		mcor[icor] .= true
		msame[isame] .= true
		mnans[inans] .= true
		mzeros[izeros] .= true
		mneg[ineg] .= true
		mstatic[istatic] .= true
		mstring[istring] .= true
		mdates[idates] .= true
		many[iany] .= true
		mremove = msame .|  mnans .| mzeros .| mstatic .| mstring .| mdates .| many
		return (; log=mlog, cor=mcor, remove=mremove, same=msame, nans=mnans, zeros=mzeros, neg=mneg, static=mstatic, string=mstring, dates=mdates,any=many)
	else
		iremove = unique(sort(vcat(isame, inans, izeros, istatic, istring, iany)))
		return (; log=ilog, cor=icor, remove=iremove, same=isame, nans=inans, zeros=izeros, neg=ineg, static=istatic, string=istring, dates=idates, any=iany)
	end
end