import NMFk
import LinearAlgebra
import Random

cs = Float64[]
ds = Float64[]
for i = 1:100
	Random.seed!(i)
	nummixtures = 20
	numbuckets = 2
	numconstituents = 2
	mixer = rand(nummixtures, numbuckets)
	for i = 1:nummixtures
		mixer[i, :] /= sum(mixer[i, :])
	end
	buckets = [1 0; 0 1]
	bucketdeltas = permutedims([1 0])
	deltaindices = Int[1]
	concentrations = mixer * buckets
	deltas = NMFk.computedeltas(mixer, buckets, bucketdeltas, deltaindices)

	#=
	@info("Concentrations")
	display(concentrations)
	@info("Deltas")
	display(deltas)
	=#

	fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2)
	fitdeltas = NMFk.computedeltas(fitmixer, fitbuckets, fitbucketdeltas, deltaindices)

	#=
	@info("Match concentrations")
	display(concentrations - fitmixer * fitbuckets)
	@info("Match deltas")
	display(deltas - fitdeltas)
	=#

	#println("c = ", sum(collect(concentrations - fitmixer * fitbuckets).^2))
	#println("d = ", sum(collect(deltas - fitdeltas).^2))
	push!(cs, sum(collect(concentrations - fitmixer * fitbuckets).^2))
	push!(ds, sum(collect(deltas - fitdeltas).^2))
	#=
	@info("Bucket concentrations")
	display(buckets)
	@info("Bucket deltas")
	display(bucketdeltas)
	@info("Bucket concentrations")
	display(fitbuckets)
	@info("Bucket deltas")
	display(fitbucketdeltas)
	=#
	#=
	@show LinearAlgebra.norm(evalgrad(fitmixer, fitbuckets, fitbucketdeltas))
	@show LinearAlgebra.norm(evalgrad(mixer, buckets, bucketdeltas))
	=#
end
@show minimum(cs)
@show minimum(ds)
