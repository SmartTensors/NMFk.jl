import MultivariateStats

function regression(P::Array{T}, Mtrain::Matrix{T}, Mpredict::Matrix{T}; method::Union{Symbol,Nothing}=:ridge, improve::Bool=true, bias::Bool=true, r::Number=0.1, showerror::Bool=false) where T
	if method == nothing
		method = :ridge
	end
	local W, H, of, sil, aic
	local Ab
	Xe = Array{T}(undef, size(Mpredict, 1), size(P, 2), size(P, 3))
	try
		al = []
		if method == :ridge
			push!(al, r)
		end
		for k = 1:size(P, 3)
			Ab = MultivariateStats.eval(method)(Mtrain, P[:,:,k], al...; trans=false, bias=bias)
			if bias
				A, b = Ab[1:end-1,:], Ab[end:end,:]
				# @show NMFk.normnan((Mtrain * A .+ b) .- P[:,:,k])
				Xe[:,:,k] = Mpredict * A .+ b
				# @show NMFk.normnan((Ab' * [Mtrain ones(size(Mtrain, 1))]')' .- P[:,:,k])
			else
				Xe[:,:,k] = Mpredict * Ab
			end
		end
	catch e
		showerror && (display(e))
		Ab = nothing
		Xe = nothing
	end
	if improve
		Xe = Array{T}(undef, size(Mpredict, 1), size(P, 2), size(P, 3))
		nk = size(Mtrain, 2)+1
		for k = 1:size(P, 3)
			if Ab == nothing
				ng = size(P, 1)
				if nk <= ng
					Ab = copy(P[1:nk,:,k])
				else
					Ab = repeat(P[:,:,k], outer=[Int(ceil(nk/ng)),1])[1:nk,:]
				end
				Ab[isnan.(Ab)] .= 1
				Wnonneg = false
			else
				Wnonneg = false
			end
			@Suppressor.suppress W, H, of, sil, aic = NMFk.execute(permutedims(P[:,:,k]), nk, 1; Wnonneg=Wnonneg, Hinit=[Mtrain ones(size(Mtrain, 1))]', Hfixed=true, method=:ipopt, regularizationweight=0.)
			# @show NMFk.normnan((W * [Mtrain ones(size(Mtrain, 1))]')' .- P[:,:,k])
			Xe[:,:,k] = permutedims(W * permutedims([Mpredict ones(size(Mpredict, 1))]))
		end
	end
	return Xe
end