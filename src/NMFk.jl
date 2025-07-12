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

global global_quiet = true
global restart = false
global imagedpi = 300
global first_warning = true

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
include("NMFkGeo.jl")
include("NMFkWells.jl")

restartoff()

# NMFk.welcome()

end