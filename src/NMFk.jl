__precompile__()

"Non-negative Matrix Factorization + k-means Clustering"
module NMFk
	import NMF
	import Distances
	import Clustering
	import JuMP
	import Ipopt
	import JLD
	import ReusableFunctions

	const nmfkdir = splitdir(splitdir(Base.source_path())[1])[1]

	quiet = true
	restart = false

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

	restartoff()
end