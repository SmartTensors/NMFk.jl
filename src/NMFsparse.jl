function NMFsparse(x::Matrix; cf::Symbol=:kl, sparsity::Int=5, max_iter::Int=100, conv_eps::Number=0.001, seed=0, k::Int=5, div_beta::Number=-1, w_ind = trues(k), h_ind = trues(k), initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), quiet::Bool=true)
	if seed != 0
		srand(seed)
	end

	if div_beta == -1
		if cf == :kl #
			div_beta = 1
			info("Sparse NMF with Kullback-Leibler divergence (beta = $(div_beta))")
		elseif cf == :ed # Euclidean distance
			div_beta = 2
			info("Sparse NMF with Euclidean divergence (beta = $(div_beta))")
		elseif cf == :is # Itakura-Saito divergence
			div_beta = 0
			info("Sparse NMF with Itakura-Saito divergence (beta = $(div_beta))")
		else
			div_beta = 1
			info("Sparse NMF with Kullback-Leibler divergence (beta = $(div_beta))")
		end
	else
		info("Sparse NMF with fractional beta divergence (beta = $(div_beta))")
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

	flr = 1e-9
	lambda = max(w * h, flr)
	last_cost = Inf

	update_h = sum(h_ind)
	update_w = sum(w_ind)

	local it, of, last_of, divergence

	for it = 1:max_iter
		if update_h > 0
			if div_beta == 1
				dph = bsxfun(+, sum(w[:,h_ind], 1)', sparsity)
				dph = max(dph, flr)
				dmh = w[:,h_ind]' * (x ./ lambda)
				h[h_ind,:] = bsxfun(\, h[h_ind,:] .* dmh, dph)
			elseif div_beta == 2
				dph = w[:,h_ind]' * lambda .+ sparsity
				dph = max(dph, flr)
				dmh = w[:,h_ind]' * x
				h[h_ind,:] = h[h_ind,:] .* dmh ./ dph
			else
				dph = w[:,h_ind]' * lambda.^(div_beta - 1) .+ sparsity
				dph = max(dph, flr)
				dmh = w[:,h_ind]' * (x .* lambda.^(div_beta - 2))
				h[h_ind,:] = h[h_ind,:] .* dmh ./ dph;
			end
			lambda = max(w * h, flr)
		end
		if update_w > 0
			if div_beta == 1
				dpw = bsxfun(+, sum(h[w_ind,:], 2)', bsxfun(*, sum((x ./ lambda) * h[w_ind,:]' .* w[:,w_ind], 1), w[:,w_ind]))
				dpw = max(dpw, flr)
				dmw = x ./ lambda * h[w_ind,:]' + bsxfun(*, sum(bsxfun(*, sum(h[w_ind,:], 2)', w[:,w_ind]), 1), w[:,w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			elseif div_beta == 2
				dpw = lambda * h[w_ind, :]' + bsxfun(*, sum(x * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				dpw = max(dpw, flr)
				dmw = x * h[w_ind, :]' + bsxfun(*, sum(lambda * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			else
				dpw = lambda.^(div_beta - 1) * h[w_ind, :]' + bsxfun(*, sum((x .* lambda.^(div_beta - 2)) * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				dpw = max(dpw, flr)
				dmw = (x .* lambda.^(div_beta - 2)) * h[w_ind, :]' + bsxfun(*, sum(lambda.^(div_beta - 1) * h[w_ind, :]' .* w[:, w_ind], 1), w[:, w_ind])
				w[:,w_ind] = w[:,w_ind] .* dmw ./ dpw
			end
			w = bsxfun(\, w, sqrt(sum(w.^2, 1)))
			lambda = max(w * h, flr)
		end
		if div_beta == 1
			divergence = sum(x .* log(x ./ lambda) - x + lambda)
		elseif div_beta == 2
			divergence = sum((x - lambda) .^ 2)
		elseif div_beta == 0
			divergence = sum(x ./ lambda - log(x ./ lambda) - 1)
		else
			divergence = sum(x.^div_beta + (div_beta - 1) * lambda.^div_beta  - div_beta * x .* lambda.^(div_beta - 1))/(div_beta * (div_beta - 1))
		end
		of = divergence + sum(h .* sparsity)

		!quiet && info("iteration $(it) divergence = $(divergence) objective function = $(of)")

		if it > 1 && conv_eps > 0
			if (abs(of - last_of) / last_of) < conv_eps
				!quiet && info("Convergence reached!")
				break
			end
		end
		last_of = of
	end
	return w, h, (of, divergence, it)
end

function bsxfun(o::Function, x, f)
	broadcast(o, f, x)
end