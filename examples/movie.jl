reload("NMFk")
srand(14)

W = rand(6, 3)
W = W .* [1 2 3]
H = rand(3, 4)
X = W * H

NMFk.plotnmf(X, W[:,sortperm(vec(sum(W, 1)))], H[sortperm(vec(sum(W, 1))),:]; filename="m643-true.png")

We, He, p, s = NMFk.execute(X, 3, 1; method=:ipopt, tolX=1e-15, tol=1e-14, seed=16, maxiter=9, movie=true, moviename="m643-frame0001.png")

Xe = We * He

NMFk.plotnmf(Xe, We[:,sortperm(vec(sum(We, 1)))], He[sortperm(vec(sum(We, 1))),:]; filename="m643-estimate.png")

srand(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
X = W * H

NMFk.plotnmf(X, W[:,sortperm(vec(sum(W, 1)))], H[sortperm(vec(sum(W, 1))),:]; filename="m2052-true.png")

# We, He, p, s = NMFk.execute(X, 2, 1; method=:ipopt, tolX=1e-4, tol=1e-14, seed=16, maxiter=40, movie=true, moviename="m2052-frame0001.png")
We, He, p, s = NMFk.execute(X, 2, 1; method=:simple, seed=2016, tol=1e-30, movie=true, moviename="m2052-frame0001.png")

Xe = We * He

NMFk.plotnmf(Xe, We[:,sortperm(vec(sum(We, 1)))], He[sortperm(vec(sum(We, 1))),:]; filename="m2052-estimate.png")