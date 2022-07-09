import NMFk
import Clustering
import RDatasets

iris = RDatasets.dataset("datasets", "iris")
NMFk.plot_dots(iris[:, :PetalLength], iris[:, :SepalLength], iris[:, :Species]; hover=iris[:, :Species])
data = float.(Matrix(iris)[:,1:4])

rkmeans = Clustering.kmeans(permutedims(data), 3)
ca = NMFk.labelassignements(rkmeans.assignments)
NMFk.plot_dots(iris[:, :PetalLength], iris[:, :SepalLength], ca; hover=iris[:, :Species])

rk, rbkmeans = NMFk.robustkmeans(permutedims(data), 2:5)
ca = NMFk.labelassignements(rbkmeans.assignments)
NMFk.plot_dots(iris[:, :PetalLength], iris[:, :SepalLength], ca; hover=iris[:, :Species])

W, H, fitquality, robustness, aic = NMFk.execute(data, 2:4)
o, lw, lh = NMFk.postprocess(3, W, H)
NMFk.plot_dots(iris[:, :PetalLength], iris[:, :SepalLength], lw[1]; hover=iris[:, :Species])