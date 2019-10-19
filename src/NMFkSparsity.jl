function NMFsparsity(X::AbstractMatrix{T}, k::Int; sparse_cf::Symbol=:kl, sparsity::Number=1, maxiter::Int=100000, tol::Number=1e-19, seed::Number=-1, sparse_div_beta::Number=-1, lambda::Number=1e-9, w_ind = trues(k), h_ind = trues(k), Winit::AbstractMatrix{T}=Array{T}(undef, 0, 0), Hinit::AbstractMatrix{T}=Array{T}(undef, 0, 0), quiet::Bool=NMFk.quiet) where {T,N}
	if seed != -1
		Random.seed!(seed)
	end

	if sparse_div_beta == -1
		if sparse_cf == :kl #
			sparse_div_beta = 1
			!quiet && @info("Sparse NMF with Kullback-Leibler divergence (beta = $(sparse_div_beta))")
		elseif sparse_cf == :ed # Euclidean distance
			sparse_div_beta = 2
			!quiet && @info("Sparse NMF with Euclidean divergence (beta = $(sparse_div_beta))")
		elseif sparse_cf == :is # Itakura-Saito divergence
			sparse_div_beta = 0
			!quiet && @info("Sparse NMF with Itakura-Saito divergence (beta = $(sparse_div_beta))")
		else
			sparse_div_beta = 1
			!quiet && @info("Sparse NMF with Kullback-Leibler divergence (beta = $(sparse_div_beta))")
		end
	else
		!quiet && @info("Sparse NMF with fractional beta divergence (beta = $(sparse_div_beta))")
	end

	(m, n) = size(X)
	if sizeof(Winit) == 0
		W = rand(m, k)
	else
		@assert (m, k) == size(Winit)
		W = Winit
	end
	if sizeof(Hinit) == 0
		H = rand(k, n)
	else
		@assert (k, n) == size(Hinit)
		H = Hinit
	end

	Wn = sqrt.(sum(W.^2; dims=1))
	W  = bsxfun(\, W, Wn)
	H  = bsxfun(*, H, Wn')

	lambda_new = max.(W * H, lambda)
	last_cost = Inf

	update_h = sum(h_ind)
	update_w = sum(w_ind)

	local it, of, last_of, divergence

	it = 0
	while it < maxiter
		it += 1
		if update_h > 0
			if sparse_div_beta == 1
				dph = bsxfun(+, sum(W[:,h_ind]; dims=1)', sparsity)
				dph = max.(dph, lambda)
				dmh = W[:,h_ind]' * (X ./ lambda_new)
				H[h_ind,:] = bsxfun(\, H[h_ind,:] .* dmh, dph)
			elseif sparse_div_beta == 2
				dph = W[:,h_ind]' * lambda_new .+ sparsity
				dph = max.(dph, lambda)
				dmh = W[:,h_ind]' * X
				H[h_ind,:] = H[h_ind,:] .* dmh ./ dph
			else
				dph = W[:,h_ind]' * lambda_new.^(sparse_div_beta - 1) .+ sparsity
				dph = max.(dph, lambda)
				dmh = W[:,h_ind]' * (X .* lambda_new.^(sparse_div_beta - 2))
				H[h_ind,:] = H[h_ind,:] .* dmh ./ dph;
			end
			lambda_new = max.(W * H, lambda)
		end
		if update_w > 0
			if sparse_div_beta == 1
				dpw = bsxfun(+, sum(H[w_ind,:]; dims=2)', bsxfun(*, sum((X ./ lambda_new) * H[w_ind,:]' .* W[:,w_ind]; dims=1), W[:,w_ind]))
				dpw = max.(dpw, lambda)
				dmw = X ./ lambda_new * H[w_ind,:]' + bsxfun(*, sum(bsxfun(*, sum(H[w_ind,:]; dims=2)', W[:,w_ind]); dims=1), W[:,w_ind])
				W[:,w_ind] = W[:,w_ind] .* dmw ./ dpw
			elseif sparse_div_beta == 2
				dpw = lambda_new * H[w_ind,:]' + bsxfun(*, sum(x * H[w_ind,:]' .* W[:,w_ind]; dims=1), W[:,w_ind])
				dpw = max.(dpw, lambda)
				dmw = X * H[w_ind,:]' + bsxfun(*, sum(lambda_new * H[w_ind,:]' .* W[:,w_ind]; dims=1), W[:,w_ind])
				W[:,w_ind] = W[:,w_ind] .* dmw ./ dpw
			else
				dpw = lambda_new.^(sparse_div_beta - 1) * H[w_ind, :]' + bsxfun(*, sum((X .* lambda_new.^(sparse_div_beta - 2)) * H[w_ind, :]' .* W[:, w_ind]; dims=1), W[:, w_ind])
				dpw = max.(dpw, lambda)
				dmw = (X .* lambda_new.^(sparse_div_beta - 2)) * H[w_ind, :]' + bsxfun(*, sum(lambda_new.^(sparse_div_beta - 1) * H[w_ind, :]' .* W[:, w_ind]; dims=1), W[:, w_ind])
				W[:,w_ind] = W[:,w_ind] .* dmw ./ dpw
			end
			W = bsxfun(\, W, sqrt.(sum(W.^2; dims=1)))
			lambda_new = max.(W * H, lambda)
		end
		if sparse_div_beta == 1
			divergence = sum(X .* log.(X ./ lambda_new) - X + lambda_new)
		elseif sparse_div_beta == 2
			divergence = sum((X - lambda_new) .^ 2)
		elseif sparse_div_beta == 0
			divergence = sum(X ./ lambda_new - log.(X ./ lambda_new) - 1)
		else
			divergence = sum(X.^sparse_div_beta + (sparse_div_beta - 1) * lambda_new.^sparse_div_beta  - sparse_div_beta * X .* lambda_new.^(sparse_div_beta - 1))/(sparse_div_beta * (sparse_div_beta - 1))
		end
		of = divergence + sum(H .* sparsity)

		!quiet && @info("Iteration $(it): divergence = $(divergence) objective function = $(of)")

		if it > 1 && tol > 0
			if (abs(of - last_of) / last_of) < tol
				!quiet && @info("Convergence reached!")
				break
			end
		end
		last_of = of
	end
	objvalue = sum((X - W * H).^2)
	return W, H, (of, objvalue, it)
end

function bsxfun(o::Function, X, f)
	broadcast(o, f, X)
end