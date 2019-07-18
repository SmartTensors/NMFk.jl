import NMFk
import NTFk
import Random

Random.seed!(14)
W = rand(6, 3)
W = W .* [1 2 3]
H = rand(3, 4)
X = W * H
NMFk.plotnmf(X, W, H; filename="movie/m643-true.png")
We, He, p, s = NMFk.execute(X, 3, 1; method=:ipopt, tolX=1e-15, tol=1e-14, seed=16, maxiter=9, movie=true, moviename="movie/m643-frame0001.png", Hinit=convert(Array{Float32,2}, H), Winit=convert(Array{Float32,2},W))
We, He, p, s = NMFk.execute(X, 2, 1; method=:ipopt, tolX=1e-6, tol=1e-8, seed=16)
Xe = We * He
NMFk.plotnmf(Xe, We, He; filename="movie/m643-estimate.png")

We, He, p, s = NMFk.execute(X, 3, 1; method=:simple, tol=1e-14, seed=16, maxiter=3, movie=true, moviename="movie/m643-frame0001.png", moviecheat=10)
NTFk.makemovie(prefix="movie/m643", cleanup=false, numberofdigits=4, movieformat="webm")

for mcheat = 1:10
	We = deepcopy(W)
	c = 1
	We .+= rand(size(We)...) .* c
	He = deepcopy(H)
	He .+= rand(size(He)...) .* c
	Xe = We * He
	NMFk.plotnmf(Xe, We, He)
end

Random.seed!(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
X = W * H
NMFk.plotnmf(X, W, H; filename="movie/m2052-true.png")
We, He, p, s = NMFk.execute(X, 2, 1; method=:ipopt, tolX=1e-4, tol=1e-14, seed=16, maxiter=40, movie=true, moviename="movie/m2052-frame0001.png", movieorder=[2,1])
We, He, p, s = NMFk.execute(X, 2, 1; method=:ipopt, tolX=1e-14, tol=1e-14, seed=16)
Xe = We * He
NMFk.plotnmf(Xe, We[:,[2,1]], He[[2,1],:]; filename="movie/m2052-estimate.png")
Wn = copy(W)
Hn = copy(H)
Xn = copy(X)
NMFk.plotnmf(X, (Wn .= NaN), (Hn .= NaN); filename="movie/m2052-trueX.png")
NMFk.plotnmf((Xn .= NaN), W, H; filename="movie/m2052-trueWH.png")

Random.seed!(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
X = W * H
NMFk.plotnmf(X, W, H; filename="movie/m2052-true.png")
We, He, p, s = NMFk.execute(X, 2, 1; method=:simple, tol=1e-2, seed=16, movie=true, moviename="movie/m2052simple-frame0001.png")
We, He, p, s = NMFk.execute(X, 2, 1; method=:simple, tol=1e-12, seed=16)
Xe = We * He
NMFk.plotnmf(Xe, We, He; filename="movie/m2052simple-estimate.png")

rMF.loaddata("test", nw=6, nc=4, ns=3; seed=14)
NMFk.plotnmf(rMF.datamatrix, rMF.truemixer, rMF.truebucket; filename="movie/s643-true.png")
rMF.execute(3; retries=1, method=:mixmatch, tol=1e-12, tolX=1e-18, seed=14, movie=true, moviename="movie/s643-frame0001.png", movieorder=[2,3,1])
NMFk.plotnmf(rMF.datamatrix, rMF.mixers[3][:,[2,3,1]], rMF.buckets[3][[2,3,1],:]; filename="movie/s643-estimate.png")