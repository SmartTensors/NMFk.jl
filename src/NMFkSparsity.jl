function NMFsparsity(X::AbstractMatrix{T}, k::Int; cost_function::Symbol=:ed, beta_divergence::Number=-1, sparsity::Number=1, maxiter::Int=100000, tol::Number=1e-19, seed::Number=-1, lambda::Number=1e-9, w_ind::AbstractVector=trues(k), h_ind::AbstractVector=trues(k), Winit::AbstractMatrix{T}=Array{T}(undef, 0, 0), Hinit::AbstractMatrix{T}=Array{T}(undef, 0, 0), quiet::Bool=NMFk.quiet) where {T <: Number}
	# inan, izero = NMFpreprocessing!(X; lambda=lambda)

	sparse_text = sparsity == 0 ? "" : "Sparse "
	if beta_divergence == -1
		if cost_function == :kl
			beta_divergence = 1
			!quiet && @info("$(sparse_text)NMF with Kullback-Leibler divergence (beta = $(beta_divergence))")
		elseif cost_function == :ed
			beta_divergence = 2
			!quiet && @info("$(sparse_text)NMF with Euclidean divergence (beta = $(beta_divergence))")
		elseif cost_function == :is
			beta_divergence = 0
			!quiet && @info("$(sparse_text)NMF with Itakura-Saito divergence (beta = $(beta_divergence))")
		else
			beta_divergence = 2
			if !quiet
				@warn("Unknown cost function: $(cost_function)")
				@info("$(sparse_text)NMF with Euclidean divergence (beta = $(beta_divergence))")
			end
		end
	else
		!quiet && @info("$(sparse_text)NMF with fractional beta divergence (beta = $(beta_divergence))")
	end

	if seed != -1
		Random.seed!(seed)
	end

	n, m = size(X)
	if sizeof(Winit) == 0
		W = rand(n, k)
	else
		@assert (n, k) == size(Winit)
		W = Winit
	end
	if sizeof(Hinit) == 0
		H = rand(k, m)
	else
		@assert (k, m) == size(Hinit)
		H = Hinit
	end

	Wn = sqrt.(sum(W .^ 2; dims=1))
	W = W ./ Wn
	H = H .* Wn'

	X_est = max.(W * H, lambda)
	last_cost = Inf

	update_h = sum(h_ind)
	update_w = sum(w_ind)

	local it, of, last_of, divergence

	it = 0
	while it < maxiter
		it += 1
		if update_h > 0
			if beta_divergence == 1
				dph = sum(W[:,h_ind]; dims=1)' .+ sparsity
				dmh = W[:,h_ind]' * (X ./ X_est)
			elseif beta_divergence == 2
				dph = W[:,h_ind]' * X_est .+ sparsity
				dmh = W[:,h_ind]' * X
			else
				dph = W[:,h_ind]' * X_est .^ (beta_divergence - 1) .+ sparsity
				dmh = W[:,h_ind]' * (X .* X_est .^ (beta_divergence - 2))
			end
			dph = max.(dph, lambda)
			H[h_ind,:] .*= dmh ./ dph
			X_est = max.(W * H, lambda)
		end
		if update_w > 0
			if beta_divergence == 1
				dpw = sum(H[w_ind,:]; dims=2)' .+ (sum((X ./ X_est) * H[w_ind,:]' .* W[:,w_ind]; dims=1) .* W[:,w_ind])
				dmw = X ./ X_est * H[w_ind,:]' + sum(sum(H[w_ind,:]; dims=2)' .* W[:,w_ind]; dims=1) .* W[:,w_ind]
			elseif beta_divergence == 2
				dpw = X_est * H[w_ind,:]' + sum(X * H[w_ind,:]' .* W[:,w_ind]; dims=1) .* W[:,w_ind]
				dmw = X * H[w_ind,:]' + sum(X_est * H[w_ind,:]' .* W[:,w_ind]; dims=1) .* W[:,w_ind]
			else
				dpw = X_est .^ (beta_divergence - 1) * H[w_ind, :]' + sum((X .* X_est .^ (beta_divergence - 2)) * H[w_ind, :]' .* W[:,w_ind]; dims=1) .* W[:,w_ind]
				dmw = (X .* X_est .^ (beta_divergence - 2)) * H[w_ind, :]' + sum(X_est .^ (beta_divergence - 1) * H[w_ind, :]' .* W[:,w_ind]; dims=1) .* W[:,w_ind]
			end
			dpw = max.(dpw, lambda)
			W[:,w_ind] .*= dmw ./ dpw
			W ./= sqrt.(sum(W .^ 2; dims=1))
			X_est = max.(W * H, lambda)
		end
		if beta_divergence == 1
			divergence = sum(X .* log.(X ./ X_est) - X + X_est)
		elseif beta_divergence == 2
			divergence = sum((X - X_est) .^ 2)
		elseif beta_divergence == 0
			divergence = sum(X ./ X_est - log.(X ./ X_est) .- 1)
		else
			divergence = sum(X .^ beta_divergence + (beta_divergence - 1) * X_est .^ beta_divergence - beta_divergence * X .* X_est .^ (beta_divergence - 1)) / (beta_divergence * (beta_divergence - 1))
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
	objvalue = sum((X - W * H) .^ 2)
	return W, H, objvalue
end