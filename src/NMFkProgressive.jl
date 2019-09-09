function progressive(X::Matrix{T}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=10; casefilename::String="progressive", kw...) where {T}
	@assert all(map(i->sum(.!isnan.(X[i, :])) > 0, 1:size(X, 1)))
	@assert all(map(i->sum(.!isnan.(X[:, i])) > 0, 1:size(X, 2)))
	# @show map(i->sum(.!isnan.(X[i, :])), 1:size(X, 1))
	# @show map(i->sum(.!isnan.(X[:, i])), 1:size(X, 2))
	window_k = Array{Int64}(undef, 0)
	for ws in windowsize
		@info("NMFk #1: $(casefilename) Window $ws")
		W, H, fitquality, robustness, aic = NMFk.execute(X[1:ws,:], nkrange, nNMF1; casefilename="$(casefilename)_$(ws)", kw...)
		kn = findlast(i->i > 0.25, robustness)
		k = (kn == nothing) ? findmax(robustness)[2] + min(nkrange...) - 1 : kn
		push!(window_k, k)
		@info("NMFk #2: $(casefilename) Window $ws: Best $k")
		if ws < size(X, 1)
			NMFk.execute(X, k, nNMF2; Hinit=convert.(Float32, H[k]), Hfixed=true, casefilename="$(casefilename)_$(ws)_all", kw...)
		end
	end
	return window_k
end

function progressive(X::Vector{Matrix{T}}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=10; casefilename::String="progressive", kw...) where {T}
	window_k = Array{Int64}(undef, 0)
	for ws in windowsize
		@info("NMFk #1: $(casefilename) Window $ws")
		normalizevector = vcat(map(i->fill(NMFk.maximumnan(X[i][1:ws,:]), ws), 1:length(X))...)
		W, H, fitquality, robustness, aic = NMFk.execute(vcat([X[i][1:ws,:] for i = 1:length(X)]...), nkrange, nNMF1; normalizevector=normalizevector,casefilename="$(casefilename)_$(ws)", kw...)
		kn = findlast(i->i > 0.25, robustness)
		k = (kn == nothing) ? findmax(robustness)[2] + min(nkrange...) - 1 : kn
		push!(window_k, k)
		@info("NMFk #2: $(casefilename) Window $ws: Best $k")
		if ws < size(X[1], 1)
			normalizevector = vcat(map(i->fill(NMFk.maximumnan(X[i]), size(X[1], 1)), 1:length(X))...)
			NMFk.execute(vcat(X...), k, nNMF2; Hinit=convert.(Float32, H[k]), normalizevector=normalizevector, Hfixed=true, casefilename="$(casefilename)_$(ws)_all", kw...)
		end
	end
	return window_k
end
