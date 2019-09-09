import DistributedArrays

function NMFmultiplicative(X::AbstractMatrix{T}, k::Int; quiet::Bool=NMFk.quiet, tol::Number=1e-19, tolOF::Number=1e-3, maxreattempts::Int=2, maxbaditers::Int=10, maxiter::Int=1000000, stopconv::Int=10000, Winit::Matrix{T}=Array{T}(undef, 0, 0), Hinit::Matrix{T}=Array{T}(undef, 0, 0), Wfixed::Bool=false, Hfixed::Bool=false, seed::Int=-1, movie::Bool=false, moviename::AbstractString="", movieorder=1:k, moviecheat::Integer=0, cheatlevel::Number=1, normalizevector::Vector{T}=Vector{T}(undef, 0)) where {T}
	if minimum(X) < 0
		error("All matrix entries must be nonnegative")
	end
	if minimum(sum(X; dims=2)) == 0
		error("All matrix entries in a row cannot be 0!")
	end
	if minimum(sum(X; dims=1)) == 0
		error("All matrix entries in a column cannot be 0!")
	end
	izero = X .<= 0
	X[izero] .= 1e-32
	inan = isnan.(X)
	if Hfixed || Wfixed
		X[inan] .= 1e-32
	else
		X[inan] .= 1e-32
	end
	if seed >= 0
		Random.seed!(seed)
	end

	n, m = size(X)

	if length(normalizevector) == n
		X ./= normalizevector
	elseif length(normalizevector) != 0
		error("Length of normalizevector does not match: $(length(normalizevector)) vs $(n)")
	end

	consold = falses(m, m)
	inc = 0

	if sizeof(Winit) == 0
		W = rand(n, k)
	else
		W = Winit
	end
	if sizeof(Hinit) == 0
		H = rand(k, m)
	else
		H = Hinit
	end

	if movie
		Xe = W * H
		frame = 1
		NMFk.plotnmf(Xe, W[:,movieorder], H[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end

	objvalue_best = Inf
	index = Array{Int}(undef, m)
	iters = 0
	baditers = 0
	reattempts = 0
	while iters < maxiter && baditers < maxbaditers && reattempts < maxreattempts
		iters += 1
		# X1 = repmat(sum(W, 1)', 1, m)
		!Hfixed && (H = H .* (permutedims(W) * (X ./ (W * H))) ./ permutedims(sum(W; dims=1)))
		if movie
			for mcheat = 1:moviecheat
				We = copy(W)
				We .+= rand(size(We)...) .* cheatlevel ./ maxiter
				He = copy(H)
				He .+= rand(size(He)...) .* cheatlevel ./ maxiter
				Xe = We * He
				frame += 1
				NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie, filename=moviename, frame=frame)
			end
			frame += 1
			Xe = W * H
			NMFk.plotnmf(Xe, W[:,movieorder], H[movieorder,:]; movie=movie, filename=moviename, frame=frame)
		end
		# X2 = repmat(sum(H, 2)', n, 1)
		!Wfixed && (W = W .* ((X ./ (W * H)) * permutedims(H)) ./ permutedims(sum(H; dims=2)))
		if movie
			for mcheat = 1:moviecheat
				We = copy(W)
				We .+= rand(size(We)...) .* cheatlevel ./ maxiter
				He = copy(H)
				He .+= rand(size(He)...) .* cheatlevel ./ maxiter
				Xe = We * He
				frame += 1
				NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie, filename=moviename, frame=frame)
			end
			frame += 1
			Xe = W * H
			NMFk.plotnmf(Xe, W[:,movieorder], H[movieorder,:]; movie=movie, filename=moviename, frame=frame)
		end
		X[inan] = (W * H)[inan]
		if mod(iters, 10) == 0
			objvalue = sum(((X - W * H)[.!inan]).^2) # Frobenius norm is sum((X - W * H).^2)^(1/2) but why bother
			if objvalue < tol
				!quiet && println("Converged by tolerance: number of iterations $(i) $(objvalue)")
				break
			end
			if objvalue < objvalue_best
				if (objvalue_best - objvalue) < tolOF
					baditers += 1
				else
					!quiet && @info("Objective function improved substantially (more than $tolOF; $objvalue < $objvalue_best); bad iteration counter reset ...")
					baditers = 0
				end
				objvalue_best = objvalue
			else
				baditers += 1
			end
			if baditers >= maxbaditers
				reattempts += 1
				if reattempts >= maxreattempts
					!quiet && @info("Objective function does not improve substantially ($objvalue)! Maximum number of reattempts ($maxreattempts) has been reached; quit!")
				end
				baditers = 0
			else
				!quiet && @info("Objective function does not improve substantially ($objvalue)! Reattempts $reattempts Bad iterations $baditers")
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
				!quiet && println("Converged by consistency: number of iterations $(i) $(inc) $(objvalue)")
				break
			end
			consold = cons
		end
	end
	for mcheat = 1:(moviecheat*2)
		We = copy(W)
		We .+= rand(size(We)...) .* cheatlevel ./ mcheat
		He = copy(H)
		He .+= rand(size(He)...) .* cheatlevel ./ mcheat
		Xe = We * He
		frame += 1
		NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	for mcheat = 1:(moviecheat*2)
		frame += 1
		NMFk.plotnmf(X, W[:,movieorder], H[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	if length(normalizevector) == n
		X .*= normalizevector
		W .*= normalizevector
	end
	X[izero] .= 0
	X[inan] .= NaN
	objvalue = sum(((X - W * H)[.!inan]).^2)
	return W, H, objvalue
end

function NMFmultiplicative(X::DistributedArrays.DArray{T,N,Array{T,N}}, k::Int; quiet::Bool=NMFk.quiet, tol::Float64=1e-19, maxiter::Int=1000000, stopconv::Int=10000, seed::Int=-1) where {T,N}
	if minimum(X) < 0
		error("All matrix entries must be nonnegative")
	end
	if minimum(sum(X; dims=2)) == 0
		error("All matrix entries in a row cannot be 0!")
	end

	if seed >= 0
		Random.seed!(seed)
	end

	n, m = size(X)

	consold = falses(m, m)
	inc = 0

	W = DistributedArrays.distribute(rand(n, k))
	H = DistributedArrays.distribute(rand(k, m))

	index = Array{Int}(undef, m)
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
			objvalue = sum(((X - W * H)).^2) # Frobenius norm is sum((X - W * H).^2)^(1/2) but why bother
			if objvalue < tol
				!quiet && println("Converged by tolerance: number of iterations $(i) $(objvalue)")
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
				!quiet && println("Converged by consistency: number of iterations $(i) $(inc) $(objvalue)")
				break
			end
			consold = cons
		end
	end
	objvalue = sum(((X - W * H)).^2)
	return W, H, objvalue
end

