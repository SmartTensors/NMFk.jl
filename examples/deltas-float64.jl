import NMFk
import LinearAlgebra
import Random

nummixtures = 20
numbuckets = 2
Random.seed!(2015)
mixer = convert(Array{Float64,2}, rand(nummixtures, numbuckets))
for i = 1:nummixtures
	mixer[i, :] /= sum(mixer[i, :])
end
buckets = convert(Array{Float64,2}, [100 0 3 50; 5 10 20 6])
bucketdeltas = convert(Array{Float64,2}, [-10 3 0; -4 5 7])
@info("Bucket concentrations")
display(buckets)
@info("Bucket deltas")
display(bucketdeltas)
deltaindices = [1, 3, 4]
concentrations = convert(Array{Float64,2}, mixer * buckets)
# concentrations[1, 1] = NaN
deltas = convert(Array{Float64,2}, NMFk.computedeltas(mixer, buckets, bucketdeltas, deltaindices))

#=
@info("Concentrations")
display(concentrations)
@info("Deltas")
display(deltas)
=#

@info("*** No transformation ...")

fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2; random=true)
fitdeltas = NMFk.computedeltas(fitmixer, fitbuckets, fitbucketdeltas, deltaindices)

#=
@info("Match concentrations")
display(concentrations - fitmixer * fitbuckets)
@info("Match deltas")
display(deltas - fitdeltas)
=#

of_c = sum(collect(concentrations - fitmixer * fitbuckets).^2)
of_d = sum(collect(deltas - fitdeltas).^2)
of_t = of_c + of_d
println("of conc   = $of_c")
println("of deltas = $of_d")
println("of total  = $of_t (check $fitquality)")
@info("Bucket concentrations")
display(fitbuckets)
@info("Bucket deltas")
display(fitbucketdeltas)

@info("*** Scaling ...")

fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2; random=true, scale=true)
fitdeltas = NMFk.computedeltas(fitmixer, fitbuckets, fitbucketdeltas, deltaindices)

#=
@info("Match concentrations")
display(concentrations - fitmixer * fitbuckets)
@info("Match deltas")
display(deltas - fitdeltas)
=#
of_c = sum(collect(concentrations - fitmixer * fitbuckets).^2)
of_d = sum(collect(deltas - fitdeltas).^2)
of_t = of_c + of_d
println("of conc   = $of_c")
println("of deltas = $of_d")
println("of total  = $of_t (check $fitquality)")
@info("Bucket concentrations")
display(fitbuckets)
@info("Bucket deltas")
display(fitbucketdeltas)

@info("*** Normalization ...")

fitmixer, fitbuckets, fitbucketdeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2; random=true, normalize=true)
fitdeltas = NMFk.computedeltas(fitmixer, fitbuckets, fitbucketdeltas, deltaindices)

#=
@info("Match concentrations")
display(concentrations - fitmixer * fitbuckets)
@info("Match deltas")
display(deltas - fitdeltas)
=#
of_c = sum(collect(concentrations - fitmixer * fitbuckets).^2)
of_d = sum(collect(deltas - fitdeltas).^2)
of_t = of_c + of_d
println("of conc   = $of_c")
println("of deltas = $of_d")
println("of total  = $of_t (check $fitquality)")
@info("Bucket concentrations")
display(fitbuckets)
@info("Bucket deltas")
display(fitbucketdeltas)