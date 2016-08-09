import NMFk
import Mads
using Base.Test

info("Reconstruction of random signals ...")
srand(2015)
nk = 3
s1 = rand(100)
s2 = rand(100)
s3 = rand(100)
S = [s1 s2 s3]
H = [[1,1,1] [0,2,1] [1,0,2] [1,2,0]]
Mads.plotseries(S, "rand_original_sources.svg", title="Original sources", name="Source", combined=false)
X = S * H
Mads.plotseries(X, "rand_mixed_signals.svg", title="Mixed signals", name="Signal", combined=false)
info("Reconstruction of random signals using NMFk ...")
Wnmf, Hnmf, pnmf, snmf = NMFk.execute(X, 20, 2)
Mads.plotseries(Wnmf, "rand_unmixed_sources_nmf.svg", title="Unmixed sources", name="Source", combined=false)
info("Reconstruction of random signals using JuMP/NLopt ...")
Wnlopt, Hnlopt, pnlopt, snlopt = NMFk.execute(X, 20, 2, mixmatch=true, mixtures=false)
Mads.plotseries(Wnlopt, "rand_unmixed_sources_nlopt.svg", title="Unmixed sources", name="Source", combined=false)

info("Reconstruction of sin signals ...")
srand(2015)
nk = 3
s1 = (sin(0.05:0.05:5)+1)/2
s2 = (sin(0.3:0.3:30)+1)/2
s3 = (sin(0.2:0.2:20)+1)/2
S = [s1 s2 s3]
H = [[1,1,1] [0,2,1] [1,0,2] [1,2,0]]
Mads.plotseries(S, "sin_original_sources.svg", title="Original sources", name="Source", combined=true)
X = S * H
Mads.plotseries(X, "sin_mixed_signals.svg", title="Mixed signals", name="Signal", combined=true)
info("Reconstruction of sin signals using NMF ...")
Wnmf, Hnmf, pnmf, snmf = NMFk.execute(X, 20, 2)
Mads.plotseries(Wnmf, "sin_unmixed_sources_nmf.svg", title="Unmixed sources", name="Source", combined=true)
Mads.plotseries(Wnmf * Hnmf, "sin_reproduced_signals_nmf.svg", title="Reproduced signals", name="Signal", combined=true)
info("Reconstruction of sin signals using JuMP/NLopt ...")
Wnlopt, Hnlopt, pnlopt, snlopt = NMFk.execute(X, 20, 2, mixmatch=true, mixtures=false)
Mads.plotseries(Wnlopt, "sin_unmixed_sources_nlopt.svg", title="Unmixed sources", name="Source", combined=true)
Mads.plotseries(Wnlopt * Hnlopt, "sin_reproduced_signals_nlopt.svg", title="Reproduced signals", name="Signal", combined=true)

info("Reconstruction of sin/rand signals ...")
srand(2015)
nk = 3
s1 = (sin(0.05:0.05:5)+1)/2
s2 = (sin(0.3:0.3:30)+1)/2
s3 = rand(100)
S = [s1 s2 s3]
H = [[1,1,1] [0,2,1] [1,0,2] [1,2,0]]
Mads.plotseries(S, "sig_original_sources.svg", title="Original sources", name="Source", combined=true)
X = S * H
Mads.plotseries(X, "sig_mixed_signals.svg", title="Mixed signals", name="Signal", combined=true)
info("Reconstruction of sin/rand signals using NMF ...")
Wnmf, Hnmf, pnmf, snmf = NMFk.execute(X, 20, 2)
Mads.plotseries(Wnmf, "sig_unmixed_sources_nmf.svg", title="Unmixed sources", name="Source", combined=true)
Mads.plotseries(Wnmf * Hnmf, "sig_reproduced_signals_nmf.svg", title="Reproduced signals", name="Signal", combined=true)
info("Reconstruction of sin/rand signals using JuMP/NLopt ...")
Wnlopt, Hnlopt, pnlopt, snlopt = NMFk.execute(X, 20, 2, mixmatch=true, mixtures=false)
Mads.plotseries(Wnlopt, "sig_unmixed_sources_nlopt.svg", title="Unmixed sources", name="Source", combined=true)
Mads.plotseries(Wnlopt * Hnlopt, "sig_reproduced_signals_nlopt.svg", title="Reproduced signals", name="Signal", combined=true)
