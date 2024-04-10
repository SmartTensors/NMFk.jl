import Statistics
import StatsBase

function checkarray(X::Array{T,N}, cutoff::Integer=0; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func) where {T <: Number, N}
	rangeentry = Array{UnitRange{Int64}}(undef, N)
	min_firstentry = Array{Int64}(undef, N)
	max_lastentry = Array{Int64}(undef, N)
	max_record_length = Array{Int64}(undef, N)
	for d = 1:N
		@info("Dimension $(d) ...")
		dd = size(X, d)
		println("Dimension $(d): size: $(dd)")
		firstentrys = Array{Int64}(undef, dd)
		lastentrys = Array{Int64}(undef, dd)
		record_length = Array{Int64}(undef, dd)
		bad_indices = Array{Int64}(undef, 0)
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
		# @info "Dimension $(d): First entry: $(firstentrys)"
		# @info "Dimension $(d): Last  entry: $(lastentrys)"
		# md[d] = length(bad_indices) > 0 ? bad_indices[1] : 0
		max_record_length[d] = mrl
		rangeentry[d] = mfe:mle
	end
	return rangeentry
end

checkarray_nans(X::Array; kw...) = checkarrayentries(X; kw...)
checkarray_zeros(X::Array; kw...) = checkarrayentries(X, i->i>0; kw...)
checkarray_count(X::Array; kw...) = checkarrayentries(X; ecount=true, kw...)

function checkarrayentries(X::Array{T,N}, func::Function=.!isnan; quiet::Bool=false, debug::Bool=false, good::Bool=false, ecount::Bool=false, cutoff::Integer=0) where {T <: Number, N}
	local flag = true
	return_indices = Vector{Vector{Int64}}(undef, N)
	for d = 1:N
		!quiet && @info("Dimension $(d) ...")
		selected_indices = Vector{Int64}(undef, 0)
		ecount && (acount = Vector{Int64}(undef, 0))
		for i = 1:size(X, d)
			nt = ntuple(k->(k == d ? i : Colon()), N)
			c = sum(func.(X[nt...]))
			ecount && (push!(acount, c))
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

function checkmatrix(df::DataFrames.DataFrame; names=names(df), kw...)
	return checkmatrix(Matrix(df), 2; names=names, kw...)
end

function checkmatrix(x::AbstractMatrix, dim=2; quiet::Bool=false, correlation_cutoff::Number=0.99, norm_cutoff::Number=0.01, skewness_cutoff::Number=1., name::AbstractString=dim == 2 ? "Column" : "Row", names::AbstractVector=["$name $i" for i=1:size(x, dim)], masks::Bool=false)
	na = size(x, dim)
	ilog = Vector{Int64}(undef, 0)
	inans = Vector{Int64}(undef, 0)
	izeros = Vector{Int64}(undef, 0)
	ineg = Vector{Int64}(undef, 0)
	iconst = Vector{Int64}(undef, 0)
	icor = Vector{Int64}(undef, 0)
	for i = 1:na
		nt = ntuple(k->(k == dim ? i : Colon()), 2)
		isn = .!isnan.(x[nt...])
		ns = ntuple(k->(k == dim ? i : isn), 2)
		v = x[ns...]
		skiplog = true
		if sum(isn) == 0
			!quiet && @info "$(names[i]) has only NaNs!"
			push!(inans, i)
		elseif sum(v) == 0
			!quiet && @info "$(names[i]) has only zeros!"
			push!(izeros, i)
		elseif any(v .< 0)
			!quiet && @info "$(names[i]) has negative values!"
			skiplog = false
			push!(ineg, i)
		elseif minimum(v) ≈ maximum(v)
			!quiet && @info "$(names[i]) is constant!"
			push!(iconst, i)
		else
			skiplog = false
			for j = i+1:na
				nt2 = ntuple(k->(k == dim ? j : Colon()), 2)
				jsn = .!isnan.(x[nt2...])
				ns2 = ntuple(k->(k == dim ? j : jsn), 2)
				v2 = x[ns2...]
				if size(v2) != size(v)
					!quiet && @warn "$(names[i]) and $(names[j]) have different number of NaN entries!"
					ijsn = isn .|| jsn
					ns = ntuple(k->(k == dim ? i : ijsn), 2)
					ns2 = ntuple(k->(k == dim ? j : ijsn), 2)
					v = x[ns...]
					v2 = x[ns2...]
				end
				if v == v2
					!quiet && @info "$(names[i]) and $(names[j]) are equivalent!"
					skiplog = true
					push!(icor, j)
				elseif Statistics.norm(v .- v2) < norm_cutoff
					!quiet && @info "$(names[i]) and $(names[j]) are very similar!"
					skiplog = true
					push!(icor, j)
				elseif (correlation = abs(Statistics.cor(v, v2))) > correlation_cutoff
					!quiet && @info "$(names[i]) and $(names[j]) are correlated $(correlation)!"
					skiplog = true
					push!(icor, j)
				end
			end
		end
		if !skiplog
			c = abs(StatsBase.skewness(v))
			if c > skewness_cutoff
				!quiet && @info "$(names[i]) is very skewed $(c); log-transformaiton recommended!"
				push!(ilog, i)
			end
		end
	end
	icor = unique(sort(icor))
	if masks
		mlog = falses(na)
		mnans = falses(na)
		mzeros = falses(na)
		mneg = falses(na)
		mconst = falses(na)
		mcor = falses(na)
		mlog[ilog] .= true
		mnans[inans] .= true
		mzeros[izeros] .= true
		mneg[ineg] .= true
		mconst[iconst] .= true
		mcor[icor] .= true
		return mlog, mnans, mzeros, mneg, mconst, mcor
	else
		return ilog, inans, izeros, ineg, iconst, icor
	end
end