# API reference

This page highlights the main supported entry points without attempting to expose every internal helper in the broad NMFk codebase.
Use Julia help mode, for example `?NMFk.execute`, for the same source-attached documentation in a REPL.

## Result and configuration types

```@docs
NMFk.ExecuteOptions
NMFk.NMFkResult
NMFk.NMFkSweepResult
```

## Factorization, resampling, and persistence

```@docs
NMFk.execute
NMFk.tensorfactorization
NMFk.bootstrapping
NMFk.bootstrapping!
NMFk.load
NMFk.save
```

## Data checks and preprocessing

```@docs
NMFk.normalizematrix_col
```

The [grid-data manual](https://github.com/SmartTensors/NMFk.jl/blob/master/docs/griddata_manual.md) documents the current `NMFk.griddata` methods.
Additional validation helpers, including `NMFk.checkmatrix`, can be inspected in Julia help mode while the curated API reference expands incrementally.

## Coordinate conversion and distance

```@docs
NMFk.latlon_to_xy
NMFk.xy_to_latlon
NMFk.haversine
```

## Structure-aware information

```@docs
NMFk.structure_information
NMFk.aggregate_lag_information
NMFk.compare_lag_information
NMFk.optimize_binning_information
NMFk.rawdata_information
NMFk.compare_rawdata_grid
```

## Visualization and notebooks

```@docs
NMFk.mapbox_contour
```

See the [Mapbox manual](https://github.com/SmartTensors/NMFk.jl/blob/master/docs/mapbox_manual.md), the [README visualization example](https://github.com/SmartTensors/NMFk.jl#examples), and the [notebooks directory](https://github.com/SmartTensors/NMFk.jl/tree/master/notebooks) for the broader legacy plotting and notebook interfaces.

## Interactive and output helpers

```@docs
NMFk.functions
NMFk.setdpi
NMFk.stdoutcaptureon
NMFk.stdoutcaptureoff
NMFk.stderrcaptureon
NMFk.stderrcaptureoff
NMFk.stdouterrcaptureon
NMFk.stdouterrcaptureoff
NMFk.quieton
NMFk.quietoff
```
