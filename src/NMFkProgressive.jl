function progressive(X::AbstractArray{T,N}, windowsize::Vector{Int64}, nkrange::AbstractRange{Int}, nNMF1::Integer=10, nNMF2::Integer=10; casefilename::String="progressive", kw...) where {T, N}
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
