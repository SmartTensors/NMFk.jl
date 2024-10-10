import DistributedArrays

function NMFpreprocessing!(X::AbstractMatrix; lambda::Number=1e-32)
	if minimum(X) < 0
		@error("All matrix entries must be nonnegative!")
		throw(ErrorException("All matrix entries must be nonnegative!"))
	end
	if first_warning
		if minimum(sum(X; dims=2)) == 0
			@warn("All matrix entries in a row should not be 0!")
		end
		if minimum(sum(X; dims=1)) == 0
			@warn("All matrix entries in a column should not be 0!")
		end
		global first_warning = false
	end
	izero = X .<= 0
	X[izero] .= lambda
	inan = isnan.(X)
	X[inan] .= lambda
	return inan, izero
end

function NMFmultiplicative(X::AbstractMatrix{T}, k::Int; weight=1, quiet::Bool=NMFk.global_quiet, tol::Number=1e-19, tolOF::Number=1e-3, lambda::Number=1e-32, maxreattempts::Int=2, maxbaditers::Int=10, maxiter::Int=1000000, stopconv::Int=1000, Wfixed::Bool=false, Hfixed::Bool=false, Winit::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), Hinit::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), seed::Int=-1, normalizevector::AbstractVector{T}=Vector{T}(undef, 0), kw...) where {T <: Number}
	inan, izero = NMFpreprocessing!(X; lambda=lambda)
	n, m = size(X)
	if length(normalizevector) == n
		X ./= normalizevector
	elseif length(normalizevector) != 0
		error("Length of normalizing vector does not match: $(length(normalizevector)) vs $(n)")
	end

	if seed >= 0
		Random.seed!(seed)
	end

	if sizeof(Winit) == 0
		W = rand(n, k)
	else
		@assert size(Winit) == (n, k)
		W = Winit
		if sum(isnan.(W)) > 0
			error("Initial values for the W matrix entries include NaNs!")
		end
	end

	if sizeof(Hinit) == 0
		H = rand(k, m)
	else
		@assert size(Hinit) == (k, m)
		H = Hinit
		if sum(isnan.(H)) > 0
			error("Initial values for the H matrix entries include NaNs!")
		end
	end

	consold = falses(m, m)
	inc = 0
	objvalue_best = Inf
	index = Vector{Int}(undef, m)
	iters = 0
	baditers = 0
	reattempts = 0
	while iters < maxiter && baditers < maxbaditers && reattempts < maxreattempts
		iters += 1
		if !Hfixed
			H = H .* (permutedims(W) * (X ./ (W * H))) ./ permutedims(sum(W; dims=1))
		end
		if !Wfixed
			W = W .* ((X ./ (W * H)) * permutedims(H)) ./ permutedims(sum(H; dims=2))
		end
		X[inan] = (W * H)[inan]
		if mod(iters, 10) == 0
			objvalue = sum((((X - W * H) .* weight)[.!inan]).^2) # Frobenius norm is sum((X - W * H).^2)^(1/2) but why bother
			if objvalue < tol
				!quiet && @info("Converged by tolerance! Number of iterations: $(iters) Objective function: $(objvalue) < $(tol)!")
				break
			end
			if objvalue < objvalue_best
				if (objvalue_best - objvalue) < tolOF
					baditers += 1
				else
					!quiet && println("Objective function improved substantially (more than $(tolOF); $(objvalue) < $(objvalue_best)); bad iteration counter reset ...")
					baditers = 0
				end
				objvalue_best = objvalue
			else
				baditers += 1
			end
			if baditers >= maxbaditers
				reattempts += 1
				if reattempts >= maxreattempts
					!quiet && @info("Objective function does not improve substantially ($(objvalue))! Maximum number of reattempts ($(maxreattempts)) has been reached; quit!")
				end
				baditers = 0
			else
				!quiet && println("Objective function does not improve substantially ($(objvalue))! Reattempts: $(reattempts) Bad iterations: $(baditers)")
			end
			H = max.(H, eps())
			W = max.(W, eps())
			for q = 1:m
				index[q] = argmin(H[:, q])
			end
			# sum(map(i->sum(index.==i).^2, 1:3))
			cons = repeat(index, 1, m) .== repeat(permutedims(index), m, 1)
			consdiff = sum(cons .!= consold)
			if consdiff == 0
				inc += 1
			else
				inc = 0
			end
			if inc > stopconv # this criteria is almost never achieved
				!quiet && @info("Converged by consistency! Number of iterations: $(iters) Objective function: $(objvalue)")
				break
			end
			consold = cons
		end
	end
	if length(normalizevector) == n
		X .*= normalizevector
		W .*= normalizevector
	end
	X[izero] .= 0
	X[inan] .= NaN
	objvalue = sum((((X - W * H) .* weight)[.!inan]).^2)
	return W, H, objvalue
end

function NMFmultiplicative(X::DistributedArrays.DArray{T,N,Array{T,N}}, k::Int; Winit::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), Hinit::AbstractMatrix{T}=Matrix{T}(undef, 0, 0), quiet::Bool=NMFk.global_quiet, tol::Float64=1e-19, lambda::Number=1e-32, maxiter::Int=1000000, stopconv::Int=10000, seed::Int=-1, kw...) where {T <: Number, N}
	inan, izero = NMFpreprocessing!(X; lambda=lambda)
	if seed >= 0
		Random.seed!(seed)
	end

	n, m = size(X)
	if sizeof(Winit) == 0
		W = DistributedArrays.distribute(rand(n, k))
	else
		@assert size(Winit) == (n, k)
		W = Winit
		if sum(isnan.(W)) > 0
			error("Initial values for the W matrix entries include NaNs!")
		end
	end

	if sizeof(Hinit) == 0
		H = DistributedArrays.distribute(rand(k, m))
	else
		@assert size(Hinit) == (k, m)
		H = Hinit
		if sum(isnan.(H)) > 0
			error("Initial values for the H matrix entries include NaNs!")
		end
	end

	consold = falses(m, m)
	inc = 0
	index = Vector{Int}(undef, m)
	for i = 1:maxiter
		a = permutedims(collect(sum(W; dims=1)))
		da = DistributedArrays.distribute(a)
		H = H .* (W' * (X ./ (W * H))) ./ da
		a = permutedims(collect(sum(H; dims=2)))
		da = DistributedArrays.distribute(a)
		HlT = permutedims(collect(H))
		HaT = DistributedArrays.distribute(HlT)
		W = W .* ((X ./ (W * H)) * HaT) ./ da
		if mod(i, 10) == 0
			objvalue = sum(((X - W * H)).^2)
			if objvalue < tol
				!quiet && @info("Converged by tolerance: number of iterations $(i) $(objvalue)")
				break
			end
			H = max.(H, eps())
			W = max.(W, eps())
			for q = 1:m
				index[q] = argmin(H[:, q])
			end
			cons = repeat(index, 1, m) .== repeat(permutedims(index), m, 1)
			consdiff = sum(cons .!= consold)
			if consdiff == 0
				inc += 1
			else
				inc = 0
			end
			if inc > stopconv # this criteria is almost never achieved
				!quiet && @info("Converged by consistency: number of iterations $(i) $(inc) $(objvalue)")
				break
			end
			consold = cons
		end
	end
	X[izero] .= 0
	X[inan] .= NaN
	objvalue = sum((X - W * H).^2)
	return W, H, objvalue
end

