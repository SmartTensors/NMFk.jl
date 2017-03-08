import ReusableFunctions

"Execute NMFk analysis for a range of number of sources (and optionally save the resutlts"
function maker3execute(commandfunction::Function)
	function make_execute_singlerun_r3(X::Matrix, nk::Int; quiet::Bool=true, ipopt::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array(Int, 0, 0), deltas::Matrix{Float32}=Array(Float32, 0, 0), deltaindices::Vector{Int}=Array(Int, 0), best::Bool=true, mixmatch::Bool=false, normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, matchwaterdeltas::Bool=false, maxiter::Int=10000, tol::Float64=1.0e-19, regularizationweight::Float32=convert(Float32, 0), ratiosweight::Float32=convert(Float32, 1), weightinverse::Bool=false, transpose::Bool=false)
		arguments = tuple(X, nk, ipopt, ratios, ratioindices, deltas, deltaindices, best, mixmatch, normalize, scale, mixtures, matchwaterdeltas, maxiter, tol, regularizationweight, ratiosweight, weightinverse, transpose)
		return ReusableFunctions.maker3function(commandfunction, arguments)
	end
end