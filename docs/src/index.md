# NMFk.jl

NMFk combines nonnegative matrix factorization with clustering and model-selection metrics to estimate the number of signals represented in a matrix.
It also provides preprocessing, persistence, visualization, and structure-aware tensor information tools used by SmartTensors workflows.

## Installation

NMFk requires Julia 1.12 or later.

```julia
import Pkg
Pkg.add("NMFk")
```

Load the package with an explicit import:

```julia
import NMFk
```

Start with the [getting-started example](getting-started.md), consult the [guides and research index](guides.md) for specialized workflows, or browse the [curated API reference](api.md).

## Documentation roles

NMFk keeps ordinary Julia docstrings with the source so they remain available through Julia's help mode.
[DocumentFunction.jl](https://github.com/madsjulia/DocumentFunction.jl) helps audit and author those static docstrings, while Documenter.jl builds and publishes this searchable site.
