import NMFk
import Random

nWells = 20
nSources = 2
nSpecies = 3
Random.seed!(2015);

W = rand(nWells, nSources)
for i = 1:nWells
	W[i, :] ./= sum(W[i, :])
end
display(W)

H = [100 0 3; 5 10 20]

X = W * H

X[1, 1] = NaN
display(X)

We, He, fit, sil, aic, kopt = NMFk.execute(X, 2:5; save=false, mixture=:mixmatch);

He[2]

H

We[2]

W

NMFk.plotmatrix(We[2])

NMFk.plotmatrix(W)

NMFk.plotmatrix(He[2])

NMFk.plotmatrix(H)
