info("start")
include(joinpath(Pkg.dir("Mads"), "src-interactive", "MadsParallel.jl"))
info("setprocs")
setprocs()

import NMFk

srand(2015)
nP = 100
nNMF = 100
a = rand(nP)
b = rand(nP)
Wtrue = [a b]
Htrue = [[1, 10, 0, 0, 1] [0, 0, 1, 5, 2]]'
X = Wtrue * Htrue
# X = [a a*10 b b*5 a+b*2]
W, H, p, s = NMFk.execute(X, 1:2, nNMF)
W[2] * H[2]