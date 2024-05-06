import MultivariateStats
import Statistics

"""
Predict B based on A and the mapping X -> Y; permuting all the matrices; old
"""
function mapping_old(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF::Integer=10; save::Bool=false, method::Symbol=:ipopt, regularizationweight::Number=1e-8, kw...) where {T <: Number}
	nk = size(X, 1)
	np = size(X, 2)
	local W1, H1, of1, sil1, aic1
	@info "Mapping matrix size: $nk x $(size(Y, 1))"
	Suppressor.@suppress W1, H1, of1, sil1, aic1 = NMFk.execute(permutedims(Y), nk, nNNF; Winit=permutedims(X), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...)
	a = NMFk.normnan(permutedims(B) .- (permutedims(A) * H1))
	vflip = NMFk.estimateflip_permutedims(X, Y, A, B)
	Xn = hcat(map(i->vflip[i] ? NMFk.flip(X[:,i]) : X[:,i], 1:np)...)
	Yn = hcat(map(i->vflip[i] ? NMFk.flip(Y[:,i]) : Y[:,i], 1:np)...)
	local W2, H2, of2, sil2, aic2
	Suppressor.@suppress W2, H2, of2, sil2, aic2 = NMFk.execute(permutedims(Yn), nk, nNNF; Winit=permutedims(Xn), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...)
	b = NMFk.normnan(permutedims(B) .- (permutedims(A) * H2))
	if a < b
		return W1, H1, of1, sil1, aic1
	else
		return W2, H2, of2, sil2, aic2
	end
end

"""
Predict B based on A and the mapping X -> Y; permuting all the matrices
"""
function mapping_permutedims(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF::Integer=10; kw...) where {T <: Number}
	W, H, of, sil, aic = mapping(permutedims(X), permutedims(Y), permutedims(A), permutedims(B), nNNF; kw...)
	return permutedims(H), permutedims(W), of, sil, aic
end

"""
Predict B based on A and the mapping X -> Y
"""
function mapping(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF::Integer=10; save::Bool=true, method::Symbol=:simple, regularizationweight::Number=1e-8, fliptest::Bool=false, kw...) where {T <: Number}
	kwx = method == :ipopt ? Dict(:regularizationweight=>regularizationweight) : Dict()
	nk = size(X, 2)
	np = size(X, 1)
	inan = isnan.(X)
	nz = sum(inan)
	if nz > 0
		@warn("Training matrix X has $nz NaNs!")
	end
	nz = sum(isnan.(A))
	if nz > 0
		@warn("Training matrix A has $nz NaNs!")
	end
	@info "Mapping matrix size: $nk x $(size(Y, 2))"
	X[inan] .= 0
	local W1, H1, of1, sil1, aic1
	Suppressor.@suppress W1, H1, of1, sil1, aic1 = NMFk.execute(Y, nk, nNNF; Winit=X, Wfixed=true, save=save, method=method, kw..., kwx...)
	iz = vec(NMFk.maximumnan(Y; dims=1) .== 0)
	H1[:, iz] .= 0
	if fliptest
		a = NMFk.normnan(B .- (A * H1))
		vflip = NMFk.estimateflip(X, Y, A, B)
		Xn = permutedims(hcat(map(i->vflip[i] ? NMFk.flip(X[i,:]) : X[i,:], 1:np)...))
		Yn = permutedims(hcat(map(i->vflip[i] ? NMFk.flip(Y[i,:]) : Y[i,:], 1:np)...))
		local W2, H2, of2, sil2, aic2
		Suppressor.@suppress W2, H2, of2, sil2, aic2 = NMFk.execute(Yn, nk, nNNF; Winit=Xn, Wfixed=true, save=save, method=method, kw..., kwx...)
		iz = vec(NMFk.maximumnan(Yn; dims=1) .== 0)
		H2[:, iz] .= 0
		b = NMFk.normnan(B .- (A * H2))
		X[inan] .= NaN
		if a < b
			W1[inan] .= NaN
			return W1, H1, of1, sil1, aic1
		else
			W2[inan] .= NaN
			return W2, H2, of2, sil2, aic2
		end
	else
		X[inan] .= NaN
		W1[inan] .= NaN
		return W1, H1, of1, sil1, aic1
	end
end