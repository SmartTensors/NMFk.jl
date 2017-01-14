GeostatInversion
================

[![GeostatInversion](http://pkg.julialang.org/badges/GeostatInversion_0.4.svg)](http://pkg.julialang.org/?pkg=GeostatInversion&ver=0.4) [![GeostatInversion](http://pkg.julialang.org/badges/GeostatInversion_0.5.svg)](http://pkg.julialang.org/?pkg=GeostatInversion&ver=0.5)

[![Build Status](https://travis-ci.org/madsjulia/GeostatInversion.jl.svg?branch=master)](https://travis-ci.org/madsjulia/GeostatInversion.jl)

[![Coverage Status](https://coveralls.io/repos/madsjulia/GeostatInversion.jl/badge.svg?branch=master)](https://coveralls.io/r/madsjulia/GeostatInversion.jl?branch=master)

This package provides methods for inverse analysis using parameter fields that are represented using geostatistical (stochastic) methods.
Currently, two geostatistical methods are implemented.
One is the Principal Component Geostatistical Approach (PCGA) proposed by [Kitanidis](http://dx.doi.org/10.1002/2013WR014630) & [Lee](http://dx.doi.org/10.1002/2014WR015483).
The other utilizes a Randomized Geostatistical Approach (RGA) that builds on PCGA.

Randomized Geostatistical Approach (RGA) references:

[O'Malley, D., Le, E., Vesselinov, V.V., Fast Geostatistical Inversion using Randomized Matrix Decompositions and Sketchings for Heterogeneous Aquifer Characterization, AGU Fall Meeting, San Francisco, CA, December 14–18, 2015.](http://adsabs.harvard.edu/abs/2015AGUFM.T31E..03O)
[Lin, Y, Le, E.B, O'Malley, D., Vesselinov, V.V., Bui-Thanh, T., Large-Scale Inverse Model Analyses Employing Fast Randomized Data Reduction, 2016.](submitted)

Two versions of PCGA are implemented in this package

- `pcgadirect`, which uses full matrices and direct solvers during iterations
- `pcgalsqr`, which uses low rank representations of the matrices combined with iterative solvers during iterations

The RGA method, `rga`, can use either of these approaches using the keyword argument. That is, by doing `rga(...; pcgafunc=GeostatInversion.pcgadirect)` or `rga(...; pcgafunc=GeostatInversion.pcgalsqr)`.

GeostatInversion is a module of MADS.

MADS
====

MADS is an open-source [Julia](http://julialang.org) code designed as an integrated high-performance computational framework performing a wide range of model-based analyses:

* Sensitivity Analysis
* Parameter Estimation
* Model Inversion and Calibration
* Uncertainty Quantification
* Model Selection and Averaging
* Decision Support

MADS utilizes adaptive rules and techniques which allows the analyses to be performed with minimum user input.
The code provides a series of alternative algorithms to perform each type of model analyses.

Documentation
=============

All the available MADS modules and functions are described at [madsjulia.github.io](http://madsjulia.github.io/Mads.jl)

Installation
============

After starting Julia, execute:

```
Pkg.add("GeostatInversion")
```

Installation of MADS behind a firewall
------------------------------

Julia uses git for package management. Add in the `.gitconfig` file in your home directory:

```
[url "https://"]
        insteadOf = git://
```

or execute:

```
git config --global url."https://".insteadOf git://
```

Set proxies:

```
export ftp_proxy=http://proxyout.<your_site>:8080
export rsync_proxy=http://proxyout.<your_site>:8080
export http_proxy=http://proxyout.<your_site>:8080
export https_proxy=http://proxyout.<your_site>:8080
export no_proxy=.<your_site>
```

For example, if you are doing this at LANL, you will need to execute the 
following lines in your bash command-line environment:

```
export ftp_proxy=http://proxyout.lanl.gov:8080
export rsync_proxy=http://proxyout.lanl.gov:8080
export http_proxy=http://proxyout.lanl.gov:8080
export https_proxy=http://proxyout.lanl.gov:8080
export no_proxy=.lanl.gov
```

MADS examples
=============

In Julia REPL, do the following commands:

`import Mads`

To explore getting-started instructions, execute:

`Mads.help()`

There are various examples located in the `examples` directory of the `Mads` repository.

For example, execute

`include(Mads.madsdir * "/../examples/contamination/contamination.jl")`

to perform various analyses related to contaminant transport, or execute

`include(Mads.madsdir * "/../examples/bigdt/bigdt.jl")`

to perform BIG-DT analysis