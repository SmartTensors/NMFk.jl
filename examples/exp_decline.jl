import NMFk
import Random

Random.seed!(2015)
s = exp.(10:-1:1) ./ 30000
a = rand(10) * .1 .+ s
b = rand(10) * .1 .+ s
c = rand(10) * .1 .+ s
d = rand(10) * .1 .+ s
e = rand(10) * .1 .+ s
W = [a b c d e]
H = rand(5, 8)

NMFk.plotmatrix(W * H; maxvalue=2., key_position=:none)
NMFk.plotmatrix(W; maxvalue=2., key_position=:none)
NMFk.plotmatrix(H; key_position=:none)