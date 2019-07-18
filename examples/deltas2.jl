import NMFk

nummixtures = 20
numbuckets = 2
numconstituents = 2
Random.seed!(2015)
mixer = rand(nummixtures-numbuckets, numbuckets)
for i = 1:nummixtures-numbuckets
	mixer[i, :] /= sum(mixer[i, :])
end
mixer = [eye(numbuckets, numbuckets); mixer]
buckets = [1 0.1; 0.1 1]
bucketdeltas = [0.1 1]'
@info("Bucket concentrations")
display(buckets)
@info("Bucket deltas")
display(bucketdeltas)
deltaindices = Int[1]
concentrations = mixer * buckets
deltas = NMFk.computedeltas(mixer, buckets, bucketdeltas, deltaindices)

#=
@info("Concentrations")
display(concentrations)
@info("Deltas")
display(deltas)
=#

#=
@info("Concentrations only")
fitmixer, fitbuckets, fitquality = NMFk.matchdata(concentrations, 2, normalize=true, Winit=mixer, Hinit=buckets, random=false)
println("a = ", fitquality)
display([concentrations fitmixer * fitbuckets])

@info("Concentrations and deltas (wrong)")
fitmixer, fitbuckets, fitquality = NMFk.matchdata([concentrations deltas], 2, normalize=false, Winit=mixer, Hinit=[buckets bucketdeltas], random=false)
println("a = ", fitquality)
display([concentrations deltas fitmixer * fitbuckets])
@info("Bucket concentrations/deltas")
display(fitbuckets)
=#

@info("Concentrations and deltas")
# fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.matchdata(concentrations, deltas, deltaindices, 2, normalize=true, Winit=mixer, Hinit=buckets, Hinitd=bucketdeltas, random=true, maxouteriters=100)
fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.matchdata(concentrations, deltas, deltaindices, 2, scale=false, random=true, maxouteriters=100)
fitdeltas = NMFk.computedeltas(fitmixer, fitbuckets, fitbucketdeltas, deltaindices)

#=
@info("Match concentrations")
display(concentrations - fitmixer * fitbuckets)
@info("Match deltas")
display(deltas - fitdeltas)
=#

println("c = ", sum(collect(concentrations - fitmixer * fitbuckets).^2))
println("d = ", sum(collect(deltas - fitdeltas).^2))
println("a = ", fitquality)
display([concentrations deltas fitmixer * fitbuckets fitdeltas])
@info("Bucket concentrations")
display(fitbuckets)
@info("Bucket deltas")
display(fitbucketdeltas)
