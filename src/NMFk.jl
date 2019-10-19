__precompile__()

"""
NMFk.jl: Nonnegative Matrix Factorization + k-means clustering
----

NMFk performs unsupervised machine learning based on matrix decomposition coupled with sparsity and nonnegativity constraints.

NMFk methodology allows for automatic identification of the optimal number of features (signals) present in two-dimensional data arrays (matrices).

The number of features is estimated automatically.
"""
module NMFk

import Pkg
using Printf
using Distributed

import Random
import Statistics
import LinearAlgebra
import NMF
import Distances
import Clustering
import Ipopt
import JuMP
import JLD2
import FileIO
import ReusableFunctions
import DocumentFunction

if Base.source_path() != nothing
	const nmfkdir = splitdir(splitdir(Base.source_path())[1])[1]
end

global quiet = true
global restart = false
global imagedpi = 300

modules = ["NMFk"]

include("NMFkHelp.jl")
include("NMFkHelpers.jl")
include("NMFkChecks.jl")
include("NMFkCluster.jl")
include("NMFkGeoChem.jl")
include("NMFkMixMatrix.jl")
include("NMFkMixTensor.jl")
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
include("NMFkPlot.jl")
include("NMFkPlotColors.jl")
include("NMFkPlotMatrix.jl")
include("NMFkCapture.jl")
include("NMFkRegression.jl")
include("NMFkMapping.jl")
include("NMFkPeaks.jl")
include("NMFkPreprocess.jl")
include("NMFkProgressive.jl")

restartoff()

NMFk.welcome()

end