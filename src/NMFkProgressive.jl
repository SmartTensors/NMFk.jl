function checkarray(X::Array{T,N}, cutoff::Integer=0; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func) where {T, N}
	md = Array{Int64}(undef, N)
	for d = 1:N
		@info("Dimension $d")
		dd = size(X, d)
		l = Array{Int64}(undef, dd)
		e = Array{Int64}(undef, 0)
		for i = 1:dd
			nt = ntuple(k->(k == d ? i : Colon()), N)
			firstentry = findfirst(funcfirst.(X[nt...]))
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
		@show e
		ir = sortperm(l)
		@show l[ir][1:15]
		@show l[ir][end-15:end]
		md[d] = length(e) > 0 ? e[1] : 0
		@show maximum(l)
	end
	return md
end

function checkarray_zeros(X::Array{T,N}) where {T, N}
	local flag = true
	for d = 1:N
		for i = 1:size(X, d)
			nt = ntuple(k->(k == d ? i : Colon()), N)
			flag = flag && sum((X[nt...] .> 0)) > 0
		end
	end
	return flag
end

function checkarray_nans(X::Array{T,N}) where {T, N}
	local flag = true
	for d = 1:N
		for i = 1:size(X, d)
			nt = ntuple(k->(k == d ? i : Colon()), N)
			flag = flag && sum(.!isnan.(X[nt...])) > 0
		end
	end
	return flag
end

function progressive(X::Matrix{T}, windowsize::Vector{Int64}, window_k::Vector{Int64}, nNMF1::Integer=10, nNMF2::Integer=10; casefilename::String="progressive", load::Bool=true, kw...) where {T}
	@assert checkarray_nans(X)
	@assert length(windowsize) == length(window_k)
	# @assert all(map(i->sum(.!isnan.(X[i, :])) > 0, 1:size(X, 1)))
	# @assert all(map(i->sum(.!isnan.(X[:, i])) > 0, 1:size(X, 2)))
	# @show map(i->sum(.!isnan.(X[i, :])), 1:size(X, 1))
	# @show map(i->sum(.!isnan.(X[:, i])), 1:size(X, 2))
	for (i, ws) in enumerate(windowsize)
		k = window_k[i]
		@info("NMFk #1: $(casefilename) Window $ws Features $k")
		W, H, fitquality, robustness, aic = NMFk.execute(X[1:ws,:], k, nNMF1; casefilename="$(casefilename)_$(ws)", load=load, kw...)
		if ws < size(X, 1)
			@info("NMFk #2: $(casefilename) Window $ws Features $k")
			NMFk.execute(X, k, nNMF2; Hinit=convert.(T, H), Hfixed=true, casefilename="$(casefilename)_$(ws)_all", load=load, kw...)
		end
	end
	return window_k
end

function progressive(X::Matrix{T}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=10; casefilename::String="progressive", load::Bool=true, kw...) where {T}
	@assert checkarray_nans(X)
	# @assert all(map(i->sum(.!isnan.(X[i, :])) > 0, 1:size(X, 1)))
	# @assert all(map(i->sum(.!isnan.(X[:, i])) > 0, 1:size(X, 2)))
	# @show map(i->sum(.!isnan.(X[i, :])), 1:size(X, 1))
	# @show map(i->sum(.!isnan.(X[:, i])), 1:size(X, 2))
	window_k = Array{Int64}(undef, 0)
	for ws in windowsize
		@info("NMFk #1: $(casefilename) Window $ws")
		W, H, fitquality, robustness, aic = NMFk.execute(X[1:ws,:], nkrange, nNMF1; casefilename="$(casefilename)_$(ws)", load=load, kw...)
		if length(nkrange) == 1
			k = nkrange[1]
		else
			kn = findlast(i->i > 0.25, robustness)
			k = (kn == nothing) ? findmax(robustness)[2] : kn
		end
		push!(window_k, k)
		@info("NMFk #2: $(casefilename) Window $ws: Best $k")
		if ws < size(X, 1)
			NMFk.execute(X, k, nNMF2; Hinit=convert.(T, H[k]), Hfixed=true, casefilename="$(casefilename)_$(ws)_all", load=load, kw...)
		end
	end
	return window_k
end

function progressive(X::Vector{Matrix{T}}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=10; casefilename::String="progressive", load::Bool=true, kw...) where {T}
	window_k = Array{Int64}(undef, 0)
	for ws in windowsize
		@info("NMFk #1: $(casefilename) Window $ws")
		normalizevector = vcat(map(i->fill(NMFk.maximumnan(X[i][1:ws,:]), ws), 1:length(X))...)
		W, H, fitquality, robustness, aic = NMFk.execute(vcat([X[i][1:ws,:] for i = 1:length(X)]...), nkrange, nNMF1; normalizevector=normalizevector,casefilename="$(casefilename)_$(ws)", load=load, kw...)
		kn = findlast(i->i > 0.25, robustness)
		k = (kn == nothing) ? findmax(robustness)[2] : kn
		push!(window_k, k)
		# global wws = 1
		# global wwe = ws
		# for i = 1:length(X)
		# 	display(X[i][1:ws,:] .- W[k][wws:wwe,:] * H[k])
		# 	wws += ws
		# 	wwe += ws
		# end
		@info("NMFk #2: $(casefilename) Window $ws: Best $k")
		if ws < size(X[1], 1)
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
