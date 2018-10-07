import MultivariateStats

function regression(X::Array{T}, Mtrain::Matrix{T}, Mpredict::Matrix{T}; method::Symbol=:ridge, bias::Bool=true, r::Number=0.1) where T
	Xe = Array{T}(size(Mpredict, 1), size(X, 2), size(X, 3))
	try
		al = []
		if method == :ridge
			push!(al, r)
		end
		for k = 1:size(X, 3)
			Aa = MultivariateStats.eval(method)(Mtrain, X[:,:,k], al...; trans=false, bias=bias)
			if bias
				A, b = Aa[1:end-1,:], Aa[end:end,:]
				Xe[:,:,k] = Mpredict * A .+ b
			else
				Xe[:,:,k] = Mpredict * Aa
			end
		end
	catch e
		display(e)
		return nothing
	end
	return Xe
end