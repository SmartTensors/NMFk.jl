import NMFk
import LinearAlgebra
import Random

Random.seed!(2015)
nWells = 20
nSources = 2
wellmixing = convert(Array{Float32,2}, rand(nWells, nSources))
for i = 1:nWells
	wellmixing[i, :] /= sum(wellmixing[i, :])
end
sourceconcentrations = convert(Array{Float32,2}, [100 0 3 50; 5 10 20 6])
sourcedeltas = convert(Array{Float32,2}, [-10 3 0; -4 5 7])
@info("Source concentrations")
display(sourceconcentrations)
@info("Source deltas")
display(sourcedeltas)
deltaindices = [1, 3, 4]
concentrations = convert(Array{Float32,2}, wellmixing * sourceconcentrations)
deltas = convert(Array{Float32,2}, NMFk.computedeltas(wellmixing, sourceconcentrations, sourcedeltas, deltaindices))

@info("Concentrations")
display(concentrations)
@info("Deltas")
display(deltas)

@info("NMFk without transformation ...")

fitwellmixing, fitsourceconcentrations, fitsourcedeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2; random=true)
fitdeltas = NMFk.computedeltas(fitwellmixing, fitsourceconcentrations, fitsourcedeltas, deltaindices)

@info("Estimated concentrations")
display(concentrations .- fitwellmixing * fitsourceconcentrations)
@info("Estimated deltas")
display(deltas .- fitdeltas)

of_c = sum(collect(concentrations - fitwellmixing * fitsourceconcentrations).^2)
of_d = sum(collect(deltas - fitdeltas).^2)
of_t = of_c + of_d
println("OF conc   = $of_c")
println("OF deltas = $of_d")
println("OF total  = $of_t (check $fitquality)")
@info("Source concentrations")
display(fitsourceconcentrations)
@info("Source deltas")
display(fitsourcedeltas)

@info("NMFk with Scaling ...")

fitwellmixing, fitsourceconcentrations, fitsourcedeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2; random=true, scale=true)
fitdeltas = NMFk.computedeltas(fitwellmixing, fitsourceconcentrations, fitsourcedeltas, deltaindices)

@info("Estimated concentrations")
display(concentrations .- fitwellmixing * fitsourceconcentrations)
@info("Estimated deltas")
display(deltas .- fitdeltas)

of_c = sum(collect(concentrations - fitwellmixing * fitsourceconcentrations).^2)
of_d = sum(collect(deltas - fitdeltas).^2)
of_t = of_c + of_d
println("OF conc   = $of_c")
println("OF deltas = $of_d")
println("OF total  = $of_t (check $fitquality)")
@info("Source concentrations")
display(fitsourceconcentrations)
@info("Source deltas")
display(fitsourcedeltas)

@info("NMFk with Normalization ...")

fitwellmixing, fitsourceconcentrations, fitsourcedeltas, fitquality = NMFk.mixmatchdeltas(concentrations, deltas, deltaindices, 2; random=true, normalize=true)
fitdeltas = NMFk.computedeltas(fitwellmixing, fitsourceconcentrations, fitsourcedeltas, deltaindices)

@info("Estimated concentrations")
display(concentrations .- fitwellmixing * fitsourceconcentrations)
@info("Estimated deltas")
display(deltas .- fitdeltas)

of_c = sum(collect(concentrations .- fitwellmixing * fitsourceconcentrations).^2)
of_d = sum(collect(deltas .- fitdeltas).^2)
of_t = of_c + of_d
println("OF conc   = $of_c")
println("OF deltas = $of_d")
println("OF total  = $of_t (check $fitquality)")
@info("Source concentrations")
display(fitsourceconcentrations)
@info("Source deltas")
display(fitsourcedeltas)