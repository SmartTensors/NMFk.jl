import Revise
import NMFk
import Mads
import Random

Random.seed!(2021)

s1 = (sin.(0.05:0.05:5) .+1) ./ 2
s2 = (sin.(0.3:0.3:30) .+ 1) ./ 2
s3 = (sin.(0.5:0.5:50) .+ 1) ./ 2
s4 = rand(100)
W = [s1 s2 s3 s4]


Mads.plotseries(W)

H = [1 5 0 0 1 1 2 1 0 2; 0 1 1 5 2 1 0 0 2 3; 3 0 0 1 0 1 0 5 4 3; 1 1 4 1 5 0 1 1 5 3]

X = W * H

Mads.plotseries(X; name="Sensors")

nkrange=2:10
We, He, fitquality, robustness, aic, kopt = NMFk.execute(X, nkrange; save=false, method=:simple);

We[kopt]

He[kopt]

Mads.plotseries(W; title="Original signals")

Mads.plotseries(We[kopt] ./ maximum(We[kopt]; dims=1); title="Reconstructed signals")

NMFk.plotmatrix(H ./ maximum(H); title="Original mixing matrix")

NMFk.plotmatrix(He[kopt] ./ maximum(He[kopt]); title="Reconstructed mixing matrix")

NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), We, He, "t" .* string.(collect(1:100)), "s" .* string.(collect(1:10)); Wcasefilename="times", Hcasefilename="sensors", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)


