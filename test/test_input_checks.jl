import Test
import NMFk

Test.@testset "Execute input_checks" begin
	Test.@testset "default casefilename for load/save" begin
		X = ones(2, 3)
		load, save, casefilename, mixture, method, algorithm, clusterWmatrix =
			NMFk.input_checks(X, true, false, "", :null, :simple, :multdiv, false)
		Test.@test load == true
		Test.@test save == false
		Test.@test casefilename == "nmfk"
		Test.@test mixture == :null
		Test.@test method == :simple
		Test.@test algorithm == :multdiv
		Test.@test clusterWmatrix == false

		load, save, casefilename, mixture, method, algorithm, clusterWmatrix =
			NMFk.input_checks(X, false, true, "", :null, :simple, :multdiv, false)
		Test.@test load == false
		Test.@test save == true
		Test.@test casefilename == "nmfk"
	end

	Test.@testset "matrix-only restriction (N>2)" begin
		X3 = ones(2, 2, 2)
		Test.@test_throws ArgumentError NMFk.input_checks(X3, false, false, "", :null, :simple, :multdiv, false)
	end

	Test.@testset "mixture forces ipopt and clusterWmatrix" begin
		X3 = ones(2, 2, 2)
		load, save, casefilename, mixture, method, algorithm, clusterWmatrix =
			NMFk.input_checks(X3, false, false, "case", :mixmatch, :simple, :multdiv, false)
		Test.@test mixture == :mixmatch
		Test.@test method == :ipopt
		Test.@test clusterWmatrix == true
		Test.@test casefilename == "case"
	end

	Test.@testset "NaNs downgrade unsupported method" begin
		X = [1.0 NaN; 2.0 3.0]
		load, save, casefilename, mixture, method, algorithm, clusterWmatrix =
			NMFk.input_checks(X, false, false, "case", :null, :nmf, :multdiv, false)
		Test.@test method == :simple
		Test.@test algorithm == :multdiv
	end

	Test.@testset "nlopt multdiv maps algorithm" begin
		X = ones(2, 2)
		_, _, _, _, method, algorithm, _ = NMFk.input_checks(X, false, false, "case", :null, :nlopt, :multdiv, false)
		Test.@test method == :nlopt
		Test.@test algorithm == :LD_SLSQP
	end

	Test.@testset "method-as-algorithm shims" begin
		X = ones(2, 2)
		_, _, _, _, method, algorithm, _ = NMFk.input_checks(X, false, false, "case", :null, :multdiv, :multdiv, false)
		Test.@test method == :nmf
		Test.@test algorithm == :multdiv

		_, _, _, _, method, algorithm, _ = NMFk.input_checks(X, false, false, "case", :null, :alspgrad, :multdiv, false)
		Test.@test method == :nmf
		Test.@test algorithm == :alspgrad
	end
end
