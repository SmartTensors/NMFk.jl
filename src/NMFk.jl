module NMFk

	import NMF
	import Distances
	import Clustering
	import JuMP
	import Ipopt
	import JLD
	import ReusableFunctions
	import PyPlot
	import Gadfly

	const nmfkdir = splitdir(splitdir(Base.source_path())[1])[1]

	quiet = true
	restart = false

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
	include("NMFkDisplay.jl")
	include("NMFkCapture.jl")

end