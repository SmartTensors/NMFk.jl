@info("start")
include(joinpath(Pkg.dir("Mads"), "src-interactive", "MadsParallel.jl"))
@info("setprocs")
setprocs()

import NMFk

Random.seed!(2015)
nP = 1000
nNMF = 100
a = rand(nP)
b = rand(nP)
Wtrue = [a b]
Htrue = [[1, 10, 0, 0, 1] [0, 0, 1, 5, 2]]'
X = Wtrue * Htrue
# X = [a a*10 b b*5 a+b*2]
W, H, p, s = NMFk.execute(X, 1:2, nNMF)
# W, H, p, s = NMFk.execute(X, 1:2, nNMF, casefilename="slurm-parallel")
W[2] * H[2]