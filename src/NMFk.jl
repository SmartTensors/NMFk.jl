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

const nmfkdir = splitdir(splitdir(Base.source_path())[1])[1]

global quiet = true
global restart = false
global imagedpi = 300

include("NMFkHelpers.jl")
include("NMFkCluster.jl")
include("NMFkGeoChem.jl")
include("NMFkMixMatrix.jl")
include("NMFkMixTensor.jl")
include("NMFkJuMP.jl")
include("NMFkMatrix.jl")
include("NMFkExecute.jl")
include("NMFkRestart.jl")
include("NMFkFinalize.jl")
include("NMFkLoad.jl")
include("NMFkBootstrap.jl")
include("NMFkSparse.jl")
include("NMFkMultiplicative.jl")
include("NMFkPlot.jl")
include("NMFkCapture.jl")
include("NMFkMultivariateStats.jl")

restartoff()

end