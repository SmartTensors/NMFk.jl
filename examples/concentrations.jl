import NMFk
import Random

nWells = 20
nSources = 2
nSpecies = 3
Random.seed!(2015)
W = rand(nWells, nSources)
for i = 1:nWells
	W[i, :] ./= sum(W[i, :]) # mixing at the wells is set to add up to one
end
H = [100 0 3; 5 10 20] # true contaminant sources
X = W * H
X[1, 1] = NaN # missing sample
We, He, fit, sil, aic, kopt = NMFk.execute(X, 2:4; mixture=:mixmatch);
display(He[2]) # estimated mixing matrix
display(H) # true mixing matrix

We, He, fit, sil, aic, kopt = NMFk.execute(convert.(Float32, X), 2:4; mixture=:mixmatch);
display(He[2]) # estimated mixing matrix
display(H) # true mixing matrix