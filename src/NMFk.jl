__precompile__()

"Non-negative Matrix Factorization + k-means Clustering and sparsity constraints"
module NMFk

import Pkg
using Printf
using Random
using Statistics
using LinearAlgebra
using Distributed

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
include("NMFkLoad.jl")
include("NMFkBootstrap.jl")
include("NMFkSparsity.jl")
include("NMFkMultiplicative.jl")
include("NMFkPlot.jl")
include("NMFkPlotColors.jl")
include("NMFkPlotMatrix.jl")
include("NMFkCapture.jl")
include("NMFkMultivariateStats.jl")
include("NMFkPeaks.jl")
include("NMFkPreprocess.jl")
include("NMFkProgressive.jl")

restartoff()

end