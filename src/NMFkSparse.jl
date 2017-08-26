function NMFsparse(x::Matrix, k::Int; sparse_cf::Symbol=:kl, sparsity::Number=1, maxiter::Int=100000, tol::Number=1e-19, seed::Number=-1, sparse_div_beta::Number=-1, lambda::Number=1e-9, w_ind = trues(k), h_ind = trues(k), initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), quiet::Bool=true)
	if seed != -1
		srand(seed)
	end

	if sparse_div_beta == -1
		if sparse_cf == :kl #
			sparse_div_beta = 1
			!quiet && info("Sparse NMF with Kullback-Leibler divergence (beta = $(sparse_div_beta))")
		elseif sparse_cf == :ed # Euclidean distance
			sparse_div_beta = 2
			!quiet && info("Sparse NMF with Euclidean divergence (beta = $(sparse_div_beta))")
		elseif sparse_cf == :is # Itakura-Saito divergence
			sparse_div_beta = 0
			!quiet && info("Sparse NMF with Itakura-Saito divergence (beta = $(sparse_div_beta))")
		else
			sparse_div_beta = 1
			!quiet && info("Sparse NMF with Kullback-Leibler divergence (beta = $(sparse_div_beta))")
		end
	else
		!quiet && info("Sparse NMF with fractional beta divergence (beta = $(sparse_div_beta))")
	end

	(m, n) = size(x)
	if sizeof(initW) == 0
		w = rand(m, k)
	else
		@assert (m, k) == size(initW)
		w = initW
	end
	if sizeof(initH) == 0
		h = rand(k, n)
	else
		@assert (k, n) == size(initH)
		w = initH
	end

	wn = sqrt(sum(w.^2, 1))
	w  = bsxfun(\, w, wn)
	h  = bsxfun(*, h, wn')

	lambda_new = max(w * h, lambda)
	last_cost = Inf

	update_h = sum(h_ind)
	update_w = sum(w_ind)

	local it, of, last_of, divergence

	for it = 1:maxiter
		if update_h > 0
			if sparse_div_beta == 1
				dph = bsxfun(+, sum(w[:,h_ind], 1)', sparsity)
				dph = max(dph, lambda)
				dmh = w[:,h_ind]' * (x ./ lambda_new)
				h[h_ind,:] = bsxfun(\, h[h_ind,:] .* dmh, dph)
			elseif sparse_div_beta == 2
				dph = w[:,h_ind]' * lambda_new .+ sparsity
				dph = max(dph, lambda)
				dmh = w[:,h_ind]' * x
				h[h_ind,:] = h[h_ind,:] .* dmh ./ dph
			else
				dph = w[:,h_ind]' * lambda_new.^(sparse_div_beta - 1) .+ sparsity
				dph = max(dph, lambda)
				dmh = w[:,h_ind]' * (x .* lambda_new.^(sparse_div_beta - 2))
				h[h_ind,:] = h[h_ind,:] .* dmh ./ dph;
			end
			lambda_new = max(w * h, lambda)
		end
		if update_w > 0
			if sparse_div_beta == 1
				dpw = bsxfun(+, sum(h[w_ind,:], 2)', bsxfun(*, sum((x ./ lambda_new) * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind]))
				dpw = max(dpw, lambda)
				dmw = x ./ lambda_new * h[w_ind,:]' + bsxfun(*, sum(bsxfun(*, sum(h[w_ind,:], 2)', w[:,w_ind]), 1), w[:,w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			elseif sparse_div_beta == 2
				dpw = lambda_new * h[w_ind,:]' + bsxfun(*, sum(x * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind])
				dpw = max(dpw, lambda)
				dmw = x * h[w_ind,:]' + bsxfun(*, sum(lambda_new * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			else
				dpw = lambda_new.^(sparse_div_beta - 1) * h[w_ind, :]' + bsxfun(*, sum((x .* lambda_new.^(sparse_div_beta - 2)) * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				dpw = max(dpw, lambda)
				dmw = (x .* lambda_new.^(sparse_div_beta - 2)) * h[w_ind, :]' + bsxfun(*, sum(lambda_new.^(sparse_div_beta - 1) * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			end
			w = bsxfun(\, w, sqrt(sum(w.^2, 1)))
			lambda_new = max(w * h, lambda)
		end
		if sparse_div_beta == 1
			divergence = sum(x .* log(x ./ lambda_new) - x + lambda_new)
		elseif sparse_div_beta == 2
			divergence = sum((x - lambda_new) .^ 2)
		elseif sparse_div_beta == 0
			divergence = sum(x ./ lambda_new - log(x ./ lambda_new) - 1)
		else
			divergence = sum(x.^sparse_div_beta + (sparse_div_beta - 1) * lambda_new.^sparse_div_beta  - sparse_div_beta * x .* lambda_new.^(sparse_div_beta - 1))/(sparse_div_beta * (sparse_div_beta - 1))
		end
		of = divergence + sum(h .* sparsity)

		!quiet && info("Iteration $(it): divergence = $(divergence) objective function = $(of)")

		if it > 1 && tol > 0
			if (abs(of - last_of) / last_of) < tol
				!quiet && info("Convergence reached!")
				break
			end
		end
		last_of = of
	end
	objvalue = sum((X - W * H).^2)
	return w, h, (of, objvalue, it)
end

function bsxfun(o::Function, x, f)
	broadcast(o, f, x)
end