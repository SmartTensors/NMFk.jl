# Getting started

The following deterministic setup creates three source signals and mixes them into five observed series.

```@example getting-started
import Random

Random.seed!(2026)
a = Random.rand(15)
b = Random.rand(15)
c = Random.rand(15)
W = [a b c]
H = [1 10 0 0 1; 0 1 1 5 2; 3 0 0 1 5]
X = W * H
size(X)
```

Estimate a factorization over candidate ranks with `NMFk.execute`.
The example disables loading and saving so it does not create result artifacts in the working directory.

```julia
import NMFk

We, He, fitquality, robustness, aic, kopt = NMFk.execute(
    X,
    2:5;
    load=false,
    save=false,
    method=:simple,
)
```

`We[kopt]` and `He[kopt]` contain the selected factorization when an optimal rank is found.
The ordering and scaling of recovered factors are not expected to match the original factors exactly.

For production analyses, set a descriptive `casefilename` and `resultdir` when saved-run reuse is desired.
NMFk validates an input hash before reusing cached decompositions, so do not discard a hash mismatch without tracing the changed input.

See the repository's [examples](https://github.com/SmartTensors/NMFk.jl/tree/master/examples), [demonstrations](https://github.com/SmartTensors/NMFk.jl/tree/master/demo), and [notebooks](https://github.com/SmartTensors/NMFk.jl/tree/master/notebooks) for larger workflows.
