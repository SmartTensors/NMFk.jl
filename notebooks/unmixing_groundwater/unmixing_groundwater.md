# NMFk: Unmixing Contaminated Groundwater

**NMFk** performs nonnegative matrix factorization with k-means clustering.

This example demonstrates how NMFk can separate hydrogeochemical concentrations observed at monitoring wells into unknown contaminant-source signatures and their mixing proportions.

This type of analysis is also called **blind source separation** or **feature extraction**.

## Setup

If NMFk is not installed, run `import Pkg; Pkg.add("NMFk")` in the Julia REPL.

```julia
import NMFk
import Random
```

Assume 20 monitoring wells, two contaminant sources, and three chemical species.

```julia
nWells::Int = 20
nSources::Int = 2
Random.seed!(2015)
```

The source-mixing coefficients at each well must add to one.

```julia
W::Matrix{Float64} = rand(nWells, nSources)
for well_index::Int in axes(W, 1)
	W[well_index, :] ./= sum(W[well_index, :])
end
display(W)
```

Each row of `W` describes how the two sources are mixed at one well.

The rows of `H` describe the concentrations of the three species in each source.

```julia
H::Matrix{Float64} = Float64[100 0 3; 5 10 20]
```

Compute the synthetic concentrations observed at the wells.

```julia
X::Matrix{Float64} = W * H
```

Mark one measurement as missing with `NaN`.

```julia
X[1, 1] = NaN
display(X)
```

## Analysis

Assuming only `X` is known, NMFk estimates the unknown source signatures and well-mixing coefficients.

Compare two and three sources because the dataset contains three measured species.

Loading and saving are disabled so rerunning the notebook does not reuse or create cached results.

```julia
results::Tuple = NMFk.execute(
	X,
	2:3;
	load=false,
	save=false,
	mixture=:mixmatch,
	serial=true,
	seed=2015,
)
We::AbstractVector = results[1]
He::AbstractVector = results[2]
fit::AbstractVector = results[3]
sil::AbstractVector = results[4]
aic::AbstractVector = results[5]
kopt::Union{Int, Nothing} = results[6]
```

`We` and `He` contain the estimated matrices for the two candidate ranks.

For this synthetic example, NMFk selects two sources.

Inspect the recovered two-source signatures and compare them with the known signatures.

```julia
He[2]
H
```

Inspect the recovered well-mixing coefficients and compare them with the known coefficients.

```julia
We[2]
W
```

The recovered and reference matrices can also be plotted.

```julia
NMFk.plotmatrix(We[2])
NMFk.plotmatrix(W)
NMFk.plotmatrix(He[2])
NMFk.plotmatrix(H)
```

NMFk recovers the number of contaminant sources, their chemical signatures, and their mixing coefficients at the wells.

Repeated NMFk runs provide a stability check for the recovered source signatures.
