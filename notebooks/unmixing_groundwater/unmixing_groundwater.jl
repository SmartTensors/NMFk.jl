import NMFk
import Random

nWells::Int = 20
nSources::Int = 2
Random.seed!(2015)

W::Matrix{Float64} = rand(nWells, nSources)
for well_index::Int in axes(W, 1)
    W[well_index, :] ./= sum(W[well_index, :])
end
display(W)

H::Matrix{Float64} = Float64[100 0 3; 5 10 20]

X::Matrix{Float64} = W * H

X[1, 1] = NaN
display(X)

results::Tuple = NMFk.execute(
    X,
    2:3;
    load=false,
    save=false,
    mixture=:mixmatch,
    serial=true,
    seed=2015
)
We::AbstractVector = results[1]
He::AbstractVector = results[2]
fit::AbstractVector = results[3]
sil::AbstractVector = results[4]
aic::AbstractVector = results[5]
kopt::Union{Int, Nothing} = results[6]

He[2]

H

We[2]

W

NMFk.plotmatrix(We[2])

NMFk.plotmatrix(W)

NMFk.plotmatrix(He[2])

NMFk.plotmatrix(H)
