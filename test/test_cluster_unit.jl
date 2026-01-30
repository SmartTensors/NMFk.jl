import Test
import NMFk
import Random

Test.@testset "Clustering utilities" begin
	Test.@testset "robustkmeans(k) returns valid assignments" begin
		Random.seed!(42)
		# columns are samples; first two columns near each other, last two near each other
		X = [
			1.0 1.1 10.0 10.1
			1.0 0.9 10.0 9.9
		]
		result = NMFk.robustkmeans(X, 2, 5; maxiter=50, tol=1e-8)
		assignments = result.assignments
		Test.@test length(assignments) == size(X, 2)
		Test.@test sort(unique(assignments)) == [1, 2]
		Test.@test size(result.centers, 2) == 2
	end

	Test.@testset "clustersolutions assigns each column exactly once" begin
		f1 = [
			1.0 0.0
			0.0 1.0
			1.0 0.0
			0.0 1.0
		]
		f2 = [
			0.0 1.0
			1.0 0.0
			0.0 1.0
			1.0 0.0
		]
		labels, centers = NMFk.clustersolutions([f1, f2], true)
		Test.@test size(labels) == (2, 2)
		Test.@test labels[:, 1] == [1, 2]
		Test.@test sort(labels[:, 2]) == [1, 2]
		Test.@test size(centers) == (2, 4)
	end
end
