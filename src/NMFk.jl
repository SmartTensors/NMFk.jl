"""
NMFk.jl: Nonnegative Matrix Factorization + k-means clustering and physics constraints
----

NMFk performs unsupervised machine learning based on matrix decomposition coupled with various constraints.

NMFk provides automatic identification of the optimal number of signals (features) present in two-dimensional data arrays (matrices).

NMFk offers visualization, pre-, and post-processing capabilities
"""
module NMFk

import Pkg
import Printf
import Distributed
import Random
import Statistics
import LinearAlgebra
import NMF
import Distances
import Clustering
import Ipopt
import JuMP
import JLD2
import ReusableFunctions
import DocumentFunction

const dir = Base.pkgdir(NMFk)

global_quiet = true
restart = false
imagedpi = 300
first_warning = true

"""
    NMFkResult

Container for results produced by `NMFk.execute(X, nk, nNMF; ...)`.

The core factorization outputs are stored in `W` and `H`, along with common
quality metrics (`fit`, `robustness`, `aic`). Additional run configuration is
captured for provenance.
"""
Base.@kwdef struct NMFkResult{TW,TH,TV,TS}
	W::TW
	H::TH
	fit::TV
	robustness::TV
	aic::TV
	nk::Int
	nNMF::Int
	sizeX::TS
	casefilename::String = ""
	resultdir::String = "."
	mixture::Symbol = :null
	method::Symbol = :simple
	algorithm::Symbol = :multdiv
	clusterWmatrix::Bool = false
	meta::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

"""
    NMFkSweepResult

Container for results produced by `NMFk.execute(X, nkrange, nNMF; ...)`.
"""
Base.@kwdef struct NMFkSweepResult{TW,TH,TV,TS}
	W::TW
	H::TH
	fitquality::TV
	robustness::TV
	aic::TV
	kopt::Union{Int,Nothing}
	nkrange::Vector{Int}
	nNMF::Int
	sizeX::TS
	casefilename::String = ""
	resultdir::String = "."
	mixture::Symbol = :null
	method::Symbol = :simple
	algorithm::Symbol = :multdiv
	clusterWmatrix::Bool = false
	meta::Dict{Symbol,Any} = Dict{Symbol,Any}()
end



modules = ["NMFk"]

include("NMFkHelp.jl")
include("NMFkHelpers.jl")
include("NMFkChecks.jl")
include("NMFkCluster.jl")
include("NMFkGeoChem.jl")
include("NMFkMixMatrix.jl")
include("NMFkMixTensor.jl")
include("NMFkTensor.jl")
include("NMFkJuMP.jl")
include("NMFkMatrix.jl")
include("NMFkExecute.jl")
include("NMFkUncertainty.jl")
include("NMFkRestart.jl")
include("NMFkFinalize.jl")
include("NMFkIO.jl")
include("NMFkBootstrap.jl")
include("NMFkSparsity.jl")
include("NMFkMultiplicative.jl")
include("NMFkMultiplicativeMovie.jl")
include("NMFkPlot.jl")
include("NMFkPlotMatch.jl")
include("NMFkMovie.jl")
include("NMFkNotebooks.jl")
include("NMFkPlotWell.jl")
include("NMFkPlotColors.jl")
include("NMFkPlotProgressBar.jl")
include("NMFkPlotMapBox.jl")
include("NMFkPlotMap.jl")
include("NMFkPlotMatrix.jl")
include("NMFkPlotCluster.jl")
include("NMFkCapture.jl")
include("NMFkRegression.jl")
include("NMFkMapping.jl")
include("NMFkPeaks.jl")
include("NMFkPreprocess.jl")
include("NMFkPostprocess.jl")
include("NMFkProgressive.jl")
include("NMFkCompress.jl")
include("NMFkGeo.jl")
include("NMFkWells.jl")

restartoff()

# NMFk.welcome()

end