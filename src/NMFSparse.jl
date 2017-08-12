function NMFsparse(x::Matrix, k::Int; cf::Symbol=:kl, sparsity::Number=1, maxiter::Int=100000, tol::Number=1e-19, seed::Number=-1, div_beta::Number=-1, lambda::Number=1e-9, w_ind = trues(k), h_ind = trues(k), initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), quiet::Bool=true)
	if seed != -1
		srand(seed)
	end

	if div_beta == -1
		if cf == :kl #
			div_beta = 1
			!quiet && info("Sparse NMF with Kullback-Leibler divergence (beta = $(div_beta))")
		elseif cf == :ed # Euclidean distance
			div_beta = 2
			!quiet && info("Sparse NMF with Euclidean divergence (beta = $(div_beta))")
		elseif cf == :is # Itakura-Saito divergence
			div_beta = 0
			!quiet && info("Sparse NMF with Itakura-Saito divergence (beta = $(div_beta))")
		else
			div_beta = 1
			!quiet && info("Sparse NMF with Kullback-Leibler divergence (beta = $(div_beta))")
		end
	else
		!quiet && info("Sparse NMF with fractional beta divergence (beta = $(div_beta))")
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
			if div_beta == 1
				dph = bsxfun(+, sum(w[:,h_ind], 1)', sparsity)
				dph = max(dph, lambda)
				dmh = w[:,h_ind]' * (x ./ lambda_new)
				h[h_ind,:] = bsxfun(\, h[h_ind,:] .* dmh, dph)
			elseif div_beta == 2
				dph = w[:,h_ind]' * lambda_new .+ sparsity
				dph = max(dph, lambda)
				dmh = w[:,h_ind]' * x
				h[h_ind,:] = h[h_ind,:] .* dmh ./ dph
			else
				dph = w[:,h_ind]' * lambda_new.^(div_beta - 1) .+ sparsity
				dph = max(dph, lambda)
				dmh = w[:,h_ind]' * (x .* lambda_new.^(div_beta - 2))
				h[h_ind,:] = h[h_ind,:] .* dmh ./ dph;
			end
			lambda_new = max(w * h, lambda)
		end
		if update_w > 0
			if div_beta == 1
				dpw = bsxfun(+, sum(h[w_ind,:], 2)', bsxfun(*, sum((x ./ lambda_new) * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind]))
				dpw = max(dpw, lambda)
				dmw = x ./ lambda_new * h[w_ind,:]' + bsxfun(*, sum(bsxfun(*, sum(h[w_ind,:], 2)', w[:,w_ind]), 1), w[:,w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			elseif div_beta == 2
				dpw = lambda_new * h[w_ind,:]' + bsxfun(*, sum(x * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind])
				dpw = max(dpw, lambda)
				dmw = x * h[w_ind,:]' + bsxfun(*, sum(lambda_new * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			else
				dpw = lambda_new.^(div_beta - 1) * h[w_ind, :]' + bsxfun(*, sum((x .* lambda_new.^(div_beta - 2)) * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				dpw = max(dpw, lambda)
				dmw = (x .* lambda_new.^(div_beta - 2)) * h[w_ind, :]' + bsxfun(*, sum(lambda_new.^(div_beta - 1) * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			end
			w = bsxfun(\, w, sqrt(sum(w.^2, 1)))
			lambda_new = max(w * h, lambda)
		end
		if div_beta == 1
			divergence = sum(x .* log(x ./ lambda_new) - x + lambda_new)
		elseif div_beta == 2
			divergence = sum((x - lambda_new) .^ 2)
		elseif div_beta == 0
			divergence = sum(x ./ lambda_new - log(x ./ lambda_new) - 1)
		else
			divergence = sum(x.^div_beta + (div_beta - 1) * lambda_new.^div_beta  - div_beta * x .* lambda_new.^(div_beta - 1))/(div_beta * (div_beta - 1))
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