import NMFk
import Random

nWells::Int = 20
nSources::Int = 2
Random.seed!(2015)
W::Matrix{Float64} = rand(nWells, nSources)
for well_index::Int in axes(W, 1)
	W[well_index, :] ./= sum(W[well_index, :]) # mixing at the wells is set to add up to one
end
H::Matrix{Float64} = Float64[100 0 3; 5 10 20] # true contaminant-source signatures
X::Matrix{Float64} = W * H
X[1, 1] = NaN # missing sample
results64::Tuple = NMFk.execute(
	X,
	2:3;
	load=false,
	save=false,
	mixture=:mixmatch,
	serial=true,
	seed=2015,
)
estimated_signatures64::AbstractVector = results64[2]
display(estimated_signatures64[2]) # estimated contaminant-source signatures
display(H) # true contaminant-source signatures

X32::Matrix{Float32} = Float32.(X)
results32::Tuple = NMFk.execute(
	X32,
	2:3;
	load=false,
	save=false,
	mixture=:mixmatch,
	serial=true,
	seed=2015,
)
estimated_signatures32::AbstractVector = results32[2]
display(estimated_signatures32[2]) # estimated contaminant-source signatures
display(Float32.(H)) # true contaminant-source signatures
