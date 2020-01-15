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
			@info "Dimension $d bad indices: $bad_indices"
		else
			@info "Dimension $(d): No bad indices!"
		end
		ir = sortperm(record_length)
		if length(record_length[ir]) > 50
			@info "Dimension $(d): Worst 15 entry counts: $(record_length[ir][1:15])"
			@info "Dimension $(d): Best 15 entry counts: $(record_length[ir][end-15:end])"
		else
			@info "Dimension $(d): Entry counts: $(record_length[ir])"
		end
		mm = maximum(record_length)
		@info "Dimension $(d): Maximum entry counts: $(mm)"
		# md[d] = length(bad_indices) > 0 ? bad_indices[1] : 0
		md[d] = mm
	end
	return md
end

checkarray_nans(X::Array; kw...) = checkarrayentries(X; kw...)
checkarray_zeros(X::Array; kw...) = checkarrayentries(X, i->i>0; kw...)
checkarray_count(X::Array, func; kw...) = checkarrayentries(X, func; ecount=true, kw...)

function checkarrayentries(X::Array{T,N}, func::Function=.!isnan; good::Bool=false, ecount::Bool=false) where {T, N}
	local flag = true
	for d = 1:N
		@info("Dimension $d ...")
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
			@info "Dimension $d count: $acount"
		else
			if length(selected_indeces) > 0
				if good
					@info "Dimension $d good indices: $selected_indeces"
				else
					@info "Dimension $d bad indices: $selected_indeces"
					# nt = ntuple(k->(k == d ? selected_indeces : Colon()), N)
					# @show nt
					# display(X[nt...])
				end
			else
				if good
					@info "Dimension $(d): No good indices!"
				else
					@info "Dimension $(d): No bad indices!"
				end
			end
		end
	end
	return flag
end