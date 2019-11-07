function checkarray(X::Array{T,N}, cutoff::Integer=0; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func) where {T, N}
	md = Array{Int64}(undef, N)
	for d = 1:N
		@info("Dimension $d ...")
		dd = size(X, d)
		l = Array{Int64}(undef, dd)
		e = Array{Int64}(undef, 0)
		for i = 1:dd
			nt = ntuple(k->(k == d ? i : Colon()), N)
			firstentry = Base.findfirst(funcfirst.(X[nt...]))
			if firstentry != nothing
				lastentry = findlast(funclast.(X[nt...]))
				if lastentry != nothing
					l[i] = lastentry - firstentry + 1
				else
					l[i] = 0
				end
			else
				l[i] = 0
			end
			if l[i] <= cutoff
				push!(e, i)
			end
		end
		@info "Bad indices in dimension $d: $e"
		ir = sortperm(l)
		@show l[ir][1:15]
		@show l[ir][end-15:end]
		md[d] = length(e) > 0 ? e[1] : 0
		@show maximum(l)
	end
	return md
end

checkarray_nans(X::Array) = checkarrayentries(X)
checkarray_zeros(X::Array) = checkarrayentries(X, i->i>0)
checkarray_count(X::Array, func) = checkarrayentries(X, func; ecount=true)

function checkarrayentries(X::Array{T,N}, func::Function=.!isnan; good::Bool=false, ecount::Bool=false) where {T, N}
	local flag = true
	for d = 1:N
		badindex = Array{Int64}(undef, 0)
		ecount && (acount = Array{Int64}(undef, 0))
		for i = 1:size(X, d)
			nt = ntuple(k->(k == d ? i : Colon()), N)
			c = sum(func.(X[nt...]))
			ecount && (push!(acount, c))
			flagi = c > 0
			if good
				flagi && push!(badindex, i)
			else
				!flagi && push!(badindex, i)
			end
			flag = flag && flagi
		end
		if ecount
			@info "Count in dimension $d: $acount"
		else
			if good
				@info "Good indices in dimension $d: $badindex"
			else
				@info "Bad indices in dimension $d: $badindex"
			end
		end
	end
	return flag
end