function progressive(X::Matrix{T}, windowsize::Int64, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::String="progressive", load::Bool=true, kw...) where {T}
	# @assert checkarray_nans(X)
	@info("NMFk #1: $(casefilename) Window $windowsize")
	W, H, fitquality, robustness, aic = NMFk.execute(X[1:windowsize,:], nkrange, nNMF1; casefilename="$(casefilename)_$(windowsize)", load=load, kw...)
	if windowsize < size(X, 1)
		robustness = Array{T}(undef, 0)
		for k in nkrange
			@info("NMFk #2: $(casefilename) Window $windowsize Features $k")
			_, _, _, r, _ = NMFk.execute(X, k, nNMF2; Hinit=convert.(T, H[k]), Hfixed=true, casefilename="$(casefilename)_$(windowsize)_all", load=load, kw...)
			push!(robustness, r)
		end
		k = getk(nkrange, robustness)
	else
		k = getk(nkrange, robustness[nkrange])
	end
	return k
end

function progressive(X::Matrix{T}, windowsize::Vector{Int64}, window_k::Vector{Int64}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::String="progressive", load::Bool=true, kw...) where {T}
	# @assert checkarray_nans(X)
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

function progressive(X::Matrix{T}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::String="progressive", load::Bool=true, kw...) where {T}
	# @assert checkarray_nans(X)
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

function progressive(X::Vector{Matrix{T}}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=nNMF1; casefilename::String="progressive", load::Bool=true, kw...) where {T}
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

function getk(nkrange::Union{AbstractRange{T1},Vector{T1}}, robustness::Vector{T2}) where {T1 <: Integer, T2 <: Number}
	@assert length(nkrange) == length(robustness)
	if length(nkrange) == 1
		k = nkrange[1]
	else
		kn = findlast(i->i > 0.25, robustness)
		kn = (kn == nothing) ? findmax(robustness)[2] : kn
		k = nkrange[kn]
	end
	return k
end

function getks(nkrange::Union{AbstractRange{T1},Vector{T1}}, robustness::Vector{T2}) where {T1 <: Integer, T2 <: Number}
	@assert length(nkrange) == length(robustness)
	if length(nkrange) == 1
		k = [nkrange[1]]
	else
		kn = findall(i->i > 0.25, robustness)
		@show kn
		if (length(kn) == 0)
			k = [nkrange[findmax(robustness)[2]]]
		else
			k = nkrange[kn]
		end
	end
	return k
end