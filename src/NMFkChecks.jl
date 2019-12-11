function checkarray(X::Array{T,N}, cutoff::Integer=0; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func) where {T, N}
	md = Array{Int64}(undef, N)
	for d = 1:N
		@info("Dimension $d ...")
		dd = size(X, d)
		record_length = Array{Int64}(undef, dd)
		bad_indices = Array{Int64}(undef, 0)
		for i = 1:dd
			nt = ntuple(k->(k == d ? i : Colon()), N)
			firstentry = Base.findfirst(funcfirst.(X[nt...]))
			if firstentry != nothing
				lastentry = findlast(funclast.(X[nt...]))
				if lastentry != nothing
					record_length[i] = lastentry - firstentry + 1
				else
					record_length[i] = 0
				end
			else
				record_length[i] = 0
			end
			if record_length[i] <= cutoff
				push!(bad_indices, i)
			end
		end
		if length(bad_indices) > 0
			@info "Bad indices in dimension $d: $bad_indices"
		else
			@info "No bad indices in dimension $d!"
		end
		ir = sortperm(record_length)
		@show record_length[ir][1:15]
		@show record_length[ir][end-15:end]
		@show maximum(record_length)
		# md[d] = length(bad_indices) > 0 ? bad_indices[1] : 0
		md[d] = maximum(record_length)
	end
	return md
end

checkarray_nans(X::Array) = checkarrayentries(X)
checkarray_zeros(X::Array) = checkarrayentries(X, i->i>0)
checkarray_count(X::Array, func) = checkarrayentries(X, func; ecount=true)

function checkarrayentries(X::Array{T,N}, func::Function=.!isnan; good::Bool=false, ecount::Bool=false) where {T, N}
	local flag = true
	for d = 1:N
		selected_indeces = Array{Int64}(undef, 0)
		ecount && (acount = Array{Int64}(undef, 0))
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
			@info "Count in dimension $d: $acount"
		else
			if length(selected_indeces) > 0
				if good
					@info "Good indices in dimension $d: $selected_indeces"
				else
					@info "Bad indices in dimension $d: $selected_indeces"
				end
			else
				if good
					@info "No good indices in dimension $d!"
				else
					@info "No bad indices in dimension $d!"
				end
			end
		end
	end
	return flag
end