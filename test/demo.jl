import NMFk
import Mads
using Base.Test

srand(2015)
s1 = rand(100)
s2 = rand(100)
S = [s1 s2]
Mads.plotseries(S, "rand_original_sources.svg", title="Original sources", name="Source", combined=false)
X = [s1+s2 s1*3+s2 s1+s2*2]
Mads.plotseries(X, "rand_mixed_signals.svg", title="Mixed signals", name="Signal", combined=false)
W, H, p, s = NMFk.execute(X, 20, 2)
Mads.plotseries(W, "rand_unmixed_sources.svg", title="Reconstruncted sources", name="Source", combined=false)

srand(2015)
s1 = exp(-(0.1:0.1:10)) * 20
s2 = 10 + sin(0.1:0.1:10) * 10
S = [s1 s2]
Mads.plotseries(S, "sig_original_sources.svg", title="Original sources", name="Source", combined=true)
X = [s1+s2 s1*3+s2 s1+s2*2]
Mads.plotseries(X, "sig_mixed_signals.svg", title="Mixed signals", name="Signal", combined=true)
W, H, p, s = NMFk.execute(X, 20, 2)
Mads.plotseries(W, "sig_unmixed_sources.svg", title="Reconstruncted sources", name="Source", combined=true)
Mads.plotseries(W * H, "sig_reproduced_signals.svg", title="Reproduced signals", name="Signal", combined=true)
