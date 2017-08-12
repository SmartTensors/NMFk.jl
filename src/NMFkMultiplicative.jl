function NMFmultiplicative(X::Array, k::Int; quiet::Bool=true, maxiter::Int=1000000, stopconv::Int=Int(ceil(10^floor(log10(maxiter)/2))), initW::Matrix{Float64}=Array{Float64}(0, 0), initH::Matrix{Float64}=Array{Float64}(0, 0), seed::Int=-1)
	if minimum(X) < 0
		error("All matrix entries must be nonnegative")
	end
	if minimum(sum(X, 2)) == 0
		error("All matrix entries in a row can be 0!")
	end

	if seed >= 0
		srand(seed)
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

	index = Array(Int, m)
	for i=1:maxiter
		# X1 = repmat(sum(W, 1)', 1, m)
		H = H .* (W' * (X ./ (W * H))) ./ sum(W, 1)'
		# X2 = repmat(sum(H, 2)', n, 1)
		W = W .* ((X ./ (W * H)) * H') ./ sum(H, 2)'
		if mod(i, 10) == 0
			H = max(H, eps())
			W = max(W, eps())
			for q = 1:m
				index[q] = indmax(H[:, q])
			end
			# sum(map(i->sum(index.==i).^2, 1:3))
			cons = repmat(index, 1, m) .== repmat(index', m, 1)
			consdiff = sum(cons .!= consold)

			if consdiff == 0
				inc += 1
			else
				inc = 0
			end

			if !quiet
				@printf("\t%d\t%d\t%d\n", i, inc, consdiff)
			end

			if inc > stopconv
				break
			end

			consold = cons
		end
	end
	E = X - W * H
	objvalue = sum(E.^2)
	return W, H, objvalue
end
