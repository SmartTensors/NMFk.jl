import NMFk

cs = Float64[]
ds = Float64[]
for i = 1:100
	nummixtures = 20
	numbuckets = 2
	numconstituents = 3
	mixer = rand(nummixtures, numbuckets)
	for i = 1:nummixtures
		mixer[i, :] /= sum(mixer[i, :])
	end
	buckets = [100 0 3 50; 5 10 20 6]
	bucketdeltas = [-10 3 0; -4 5 7]
	deltaindices = [1, 3, 4]
	concentrations = mixer * buckets
	deltas = NMFk.computedeltas(mixer, buckets, bucketdeltas, deltaindices)

	#=
	@info("Concentrations")
	display(concentrations)
	@info("Deltas")
	display(deltas)
	=#

	fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.matchdata(concentrations, deltas, deltaindices, 2; deltaweightsfactor=1, verbosity=1)
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
	@show norm(evalgrad(fitmixer, fitbuckets, fitbucketdeltas))
	@show norm(evalgrad(mixer, buckets, bucketdeltas))
	=#
end
@show minimum(cs)
@show minimum(ds)
