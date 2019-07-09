import DistributedArrays

function NMFmultiplicative(X::AbstractMatrix, k::Int; quiet::Bool=NMFk.quiet, tol::Float64=1e-19, maxiter::Int=1000000, stopconv::Int=10000, initW::Matrix{Float64}=Array{Float64}(undef, 0, 0), initH::Matrix{Float64}=Array{Float64}(undef, 0, 0), seed::Int=-1, movie::Bool=false, moviename::AbstractString="", movieorder=1:k, moviecheat::Integer=0, cheatlevel::Number=1)
	if minimum(X) < 0
		error("All matrix entries must be nonnegative")
	end
	if minimum(sum(X; dims=2)) == 0
		error("All matrix entries in a row cannot be 0!")
	end
	inan = isnan.(X)
	X[inan] .= 0

	if seed >= 0
		Random.seed!(seed)
	end

	n, m = size(X)

	consold = falses(m, m)
	inc = 0

	if sizeof(initW) == 0
		W = rand(n, k)
	else
		W = initW
	end
	if sizeof(initH) == 0
		H = rand(k, m)
	else
		H = initH
	end

	if movie
		Xe = W * H
		frame = 1
		NMFk.plotnmf(Xe, W[:,movieorder], H[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end

	# maxinc = 0
	index = Array{Int}(undef, m)
	for i = 1:maxiter
		# X1 = repmat(sum(W, 1)', 1, m)
		H = H .* (permutedims(W) * (X ./ (W * H))) ./ permutedims(sum(W; dims=1))
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
		W = W .* ((X ./ (W * H)) * permutedims(H)) ./ permutedims(sum(H; dims=2))
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
		if mod(i, 10) == 0
			objvalue = sum(((X - W * H)[.!inan]).^2) # Frobenius norm is sum((X - W * H).^2)^(1/2) but why bother
			if objvalue < tol
				!quiet && println("Converged by tolerance: number of iterations $(i) $(objvalue)")
				break
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
			#if inc > maxinc
			#	maxinc = inc
			#end
			# @printf("\t%d\t%d\t%d\n", i, inc, consdiff)
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

	# maxinc = 0
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
	objvalue = sum(((X - W * H)).^2)
	return W, H, objvalue
end

