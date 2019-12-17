import MultivariateStats
import Statistics

"""
Predict B based on A and the mapping X -> Y; permuting all the matrices; old
"""
function mapping_old(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF=10; save=false, method=:ipopt, regularizationweight=1e-8, kw...) where T
	nk = size(X, 1)
	np = size(X, 2)
	local W1, H1, of1, sil1, aic1
	@info "Mapping matrix size will be: $nk x $(size(Y, 1))"
	@Suppressor.suppress W1, H1, of1, sil1, aic1 = NMFk.execute(permutedims(Y), nk, nNNF; Winit=permutedims(X), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...)
	a = NMFk.normnan(permutedims(B) .- (permutedims(A) * H1))
	vflip = NMFk.estimateflip_permutedims(X, Y, A, B)
	Xn = hcat(map(i->vflip[i] ? NMFk.flip(X[:,i]) : X[:,i], 1:np)...)
	Yn = hcat(map(i->vflip[i] ? NMFk.flip(Y[:,i]) : Y[:,i], 1:np)...)
	local W2, H2, of2, sil2, aic2
	@Suppressor.suppress W2, H2, of2, sil2, aic2 = NMFk.execute(permutedims(Yn), nk, nNNF; Winit=permutedims(Xn), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...)
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
function mapping_permutedims(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF=10; kw...) where T
	mapping(permutedims(X), permutedims(Y), permutedims(A), permutedims(B), nNNF; kw...)
end

"""
Predict B based on A and the mapping X -> Y
"""
function mapping(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF=10; save=false, method=:simple, regularizationweight=1e-8, fliptest=false, kw...) where T
	kwx = method == :ipopt ? Dict("regularizationweight"=>regularizationweight) : Dict()
	nk = size(X, 2)
	np = size(X, 1)
	nz = sum(isnan.(X))
	if nz > 0
		@warn("Training matrix X has $nz NaNs!")
	end
	nz = sum(isnan.(A))
	if nz > 0
		@warn("Training matrix A has $nz NaNs!")
	end
	@info "Mapping matrix size will be: $nk x $(size(Y, 2))"
	local W1, H1, of1, sil1, aic1
	@Suppressor.suppress W1, H1, of1, sil1, aic1 = NMFk.execute(Y, nk, nNNF; Winit=X, Wfixed=true, save=save, method=method, kw..., kwx...)
	if fliptest
		a = NMFk.normnan(B .- (A * H1))
		vflip = NMFk.estimateflip(X, Y, A, B)
		Xn = permutedims(hcat(map(i->vflip[i] ? NMFk.flip(X[i,:]) : X[i,:], 1:np)...))
		Yn = permutedims(hcat(map(i->vflip[i] ? NMFk.flip(Y[i,:]) : Y[i,:], 1:np)...))
		local W2, H2, of2, sil2, aic2
		@Suppressor.suppress W2, H2, of2, sil2, aic2 = NMFk.execute(Yn, nk, nNNF; Winit=Xn, Wfixed=true, save=save, method=method, kw..., kwx...)
		b = NMFk.normnan(B .- (A * H2))
		if a < b
			return W1, H1, of1, sil1, aic1
		else
			return W2, H2, of2, sil2, aic2
		end
	else
		return W1, H1, of1, sil1, aic1
	end
end