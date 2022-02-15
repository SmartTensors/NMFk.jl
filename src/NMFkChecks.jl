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
			if firstentry !== nothing
				firstentrys[i] = firstentry
				lastentry = Base.findlast(funclast.(ix[firstentry:end]))
				if lastentry !== nothing
					lastentrys[i] = lastentry + firstentry - 1
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
			println("Dimension $(d) bad indices: $bad_indices")
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

function checkarrayentries(X::Array{T,N}, func::Function=.!isnan; quiet::Bool=false, good::Bool=false, ecount::Bool=false) where {T <: Number, N}
	local flag = true
	return_selected_indeces = Vector{Vector{Int64}}(undef, N)
	for d = 1:N
		!quiet && @info("Dimension $(d) ...")
		selected_indeces = Vector{Int64}(undef, 0)
		ecount && (acount = Vector{Int64}(undef, 0))
		for i = 1:size(X, d)
			nt = ntuple(k->(k == d ? i : Colon()), N)
			c = sum(func.(X[nt...]))
			ecount && (push!(acount, c))
			flagi = c > 0
			if good
				flagi && push!(selected_indeces, i)
			else
				!flagi && push!(selected_indeces, i)
			end
			flag = flag && flagi
		end
		if ecount
			!quiet && println("Dimension $(d) count: $acount")
		else
			if !quiet
				if length(selected_indeces) > 0
					if good
						println("Dimension $(d) good indices: $selected_indeces")
					else
						println("Dimension $(d) bad indices: $selected_indeces")
						# nt = ntuple(k->(k == d ? selected_indeces : Colon()), N)
						# @show nt
						# display(X[nt...])
					end
				else
					if good
						println("Dimension $(d): No good indices!")
					else
						println("Dimension $(d): No bad indices!")
					end
				end
			end
		end
		return_selected_indeces[d] = selected_indeces
	end
	return return_selected_indeces
end

checkcols(x::AbstractMatrix; kw...) = checkmatrix(x::AbstractMatrix, 2; kw...)
checkrows(x::AbstractMatrix; kw...) = checkmatrix(x::AbstractMatrix, 1; kw...)

function checkmatrix(x::AbstractMatrix, dim=2; quiet::Bool=false)
	na = size(x, dim)
	name = dim == 2 ? "column" : "row"
	inans = Vector{Int64}(undef, 0)
	izeros = Vector{Int64}(undef, 0)
	ineg = Vector{Int64}(undef, 0)
	iconst = Vector{Int64}(undef, 0)
	icor = Vector{Int64}(undef, 0)
	ilog = Vector{Int64}(undef, 0)
	for i = 1:na
		nt = ntuple(k->(k == dim ? i : Colon()), 2)
		isn = .!isnan.(x[nt...])
		ns = ntuple(k->(k == dim ? i : isn), 2)
		v = x[ns...]
		skiplog = true
		if sum(isn) == 0
			!quiet && @info "Matrix $name $i has only NaNs!"
			push!(inans, i)
		elseif sum(v) == 0
			!quiet && @info "Matrix $name $i has only zeros!"
			push!(izeros, i)
		elseif any(v .< 0)
			!quiet && @info "Matrix $name $i has negative values!"
			skiplog = false
			push!(ineg, i)
		elseif minimum(v) ≈ maximum(v)
			!quiet && @info "Matrix $name $i is constant!"
			push!(iconst, i)
		else
			for j = i+1:na
				nt2 = ntuple(k->(k == dim ? j : Colon()), 2)
				jsn = .!isnan.(x[nt2...])
				ns2 = ntuple(k->(k == dim ? j : jsn), 2)
				v2 = x[ns2...]
				c = Statistics.cor(v, v2)
				if c ≈ 1
					!quiet && @info "Matrix $name $i and $name $j are correlated!"
					push!(icor, j)
				end
			end
			skiplog = false
		end
		if !skiplog
			c = abs(StatsBase.skewness(v))
			if c > 1
				!quiet && @info "Matrix $name $i is very skewed; log-transformaiton recommended!"
				push!(ilog, i)
			end
		end
	end
	return inans, izeros, ineg, iconst, unique(sort(icor)), ilog
end