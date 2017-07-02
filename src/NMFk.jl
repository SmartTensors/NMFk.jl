module NMFk

import NMF
import Distances
import Clustering
import JuMP
import Ipopt
import JLD
import ReusableFunctions

restart = false

include("NMFkCluster.jl")
include("NMFkGeoChem.jl")
include("NMFkMixMatch.jl")
include("NMFkIpopt.jl")
include("NMFkMatrix.jl")
include("NMFkExecute.jl")
include("NMFkRestart.jl")
include("NMFkFinalize.jl")
include("NMFkLoad.jl")
include("NMFSparse.jl")

end