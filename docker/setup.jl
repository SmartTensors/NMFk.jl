import Pkg
Pkg.add("NMFk")
Pkg.add("Mads")
Pkg.add("Revise")
Pkg.add("Flux")
Pkg.add("Compose")
Pkg.add("Cairo")
Pkg.add("Fontconfig")
Pkg.add("Gadfly")

import NMFk
import Mads
import Revise
import Flux
import Compose
import Cairo
import Fontconfig
import Gadfly

NMFk.test()

a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
X = W * H
X = [a a*10 b b*5 a+b*2]

nkrange = 2:3

W, H, fitquality, robustness, aic = NMFk.execute(X, nkrange; resultdir="NMFk-test-results", load=true)
resultdirpost = "NMFk-test-results/postprocessing"
figuredirpost = "NMFk-test-results/postprocessing"
NMFk.plot_feature_selecton(nkrange, fitquality, robustness; figuredir=figuredirpost)
NMFk.clusterresults(NMFk.getks(nkrange, robustness[nkrange]), W, H, string.(collect(1:20)), string.(collect(1:5)); resultdir=resultdirpost, figuredir=figuredirpost, Wcasefilename="times", Hcasefilename="attributes", biplotcolor=:WH, sortmag=false, biplotlabel=:H, point_size_nolabel=2Gadfly.pt, point_size_label=4Gadfly.pt)
