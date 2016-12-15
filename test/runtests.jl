using NMFk
using Base.Test

srand(2015)
a = rand(20)
b = rand(20)
X = convert(Array{Float32, 2}, [a a*10 b b*5 a+b*2])
W, H, p, s = NMFk.execute(X, 20, 2)
@test_approx_eq_eps p 0 1e-3
@test_approx_eq_eps s 1 1e-3
@test_approx_eq_eps H[1,2] / H[1,1] 10 1e-3
@test_approx_eq_eps H[1,5] / H[1,1] 1 1e-0
@test_approx_eq_eps H[2,4] / H[2,3] 5 1e-2
@test_approx_eq_eps H[2,5] / H[2,3] 2 1e-0

srand(2015)
a = exp(-(0:.5:10))*100
b = 100 + sin(0:20)*10
X = convert(Array{Float32, 2}, [a a*10 b b*5 a+b*2])
W, H, p, s = NMFk.execute(X, 20, 2)
@test_approx_eq_eps p 0.6 1e-0
@test_approx_eq_eps s 1 1e-3
@test_approx_eq_eps H[2,2] / H[2,1] 10 1e-3
@test_approx_eq_eps H[2,5] / H[2,1] 1 1e-0
@test_approx_eq_eps H[1,4] / H[1,3] 5 1e-2
@test_approx_eq_eps H[1,5] / H[1,3] 2 1e-0
