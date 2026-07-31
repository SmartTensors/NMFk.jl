import Test
import NMFk
import Random
import Logging

Test.@testset "Execute smoke tests" begin
    Test.@testset "execute warns when the input matrix is not normalized" begin
        mktempdir() do normalized_directory::String
            normalized_matrix::Matrix{Float64} = [0.0 0.5 1.0; 0.25 0.75 NaN; 1.0 0.0 0.5]
            Test.@test_logs min_level=Logging.Warn NMFk.execute(normalized_matrix, 2, 1; loadonly=true, load=true, save=false, casefilename="normalized", resultdir=normalized_directory, quiet=true)
        end

        mktempdir() do unnormalized_directory::String
            unnormalized_matrix::Matrix{Float64} = [0.0 0.5 1.0; 0.25 2.0 NaN; 1.0 0.0 0.5]
            Test.@test_logs (:warn, r"Input matrix is not normalized") min_level=Logging.Warn NMFk.execute(unnormalized_matrix, 2, 1; loadonly=true, load=true, save=false, casefilename="unnormalized", resultdir=unnormalized_directory, quiet=true)
        end
    end

    Test.@testset "execute_singlerun (simple) returns sane shapes" begin
        Random.seed!(123)
        X = abs.(randn(5, 4))
        W, H, obj = NMFk.execute_singlerun(X, 2; quiet=true, method=:simple, maxiter=50, tol=1e-8)
        Test.@test size(W) == (size(X, 1), 2)
        Test.@test size(H) == (2, size(X, 2))
        Test.@test isfinite(obj)
        Test.@test all(isfinite, W)
        Test.@test all(isfinite, H)
        Test.@test all(W .>= 0)
        Test.@test all(H .>= 0)
        # default normalization in execute_singlerun_compute (clusterWmatrix=false) normalizes H rows
        Test.@test isapprox(sum(@view(H[1, :])), 1.0; atol=1e-4)
        Test.@test isapprox(sum(@view(H[2, :])), 1.0; atol=1e-4)
    end

    Test.@testset "execute_run (nk=1) stays lightweight" begin
        Random.seed!(321)
        X::Matrix{Float64} = abs.(randn(6, 5))
        cancel_check_count::Base.RefValue{Int} = Ref{Int}(0)
        cancel_check::Function = function ()
            cancel_check_count[] += 1
            return nothing
        end
        run_result::Tuple = NMFk.execute_run(
            X,
            1,
            2;
            serial=true,
            veryquiet=true,
            best=true,
            maxiter=40,
            tol=1e-8,
            cancel_check=cancel_check
        )
        Wa::Matrix{Float64} = run_result[1]
        Ha::Matrix{Float64} = run_result[2]
        phi::Float64 = run_result[3]
        minsilhouette::Float64 = run_result[4]
        aic::Float64 = run_result[5]
        Test.@test size(Wa) == (size(X, 1), 1)
        Test.@test size(Ha) == (1, size(X, 2))
        Test.@test isfinite(phi)
        Test.@test isfinite(aic)
        Test.@test minsilhouette == 1
        Test.@test cancel_check_count[] == 3

        interrupt_check_count::Base.RefValue{Int} = Ref{Int}(0)
        interrupt_check::Function = function ()
            interrupt_check_count[] += 1
            throw(InterruptException())
        end
        Test.@test_throws InterruptException NMFk.execute_run(
            X,
            1,
            2;
            serial=true,
            veryquiet=true,
            maxiter=40,
            tol=1e-8,
            cancel_check=interrupt_check
        )
        Test.@test interrupt_check_count[] == 1
    end

    Test.@testset "matrix MixMatch handles missing data and selects rank" begin
        mktempdir() do result_directory::String
            Random.seed!(2015)
            reference_mixtures::Matrix{Float64} = rand(20, 2)
            for well_index::Int in axes(reference_mixtures, 1)
                reference_mixtures[well_index, :] ./= sum(reference_mixtures[well_index, :])
            end
            reference_signatures::Matrix{Float64} = Float64[100 0 3; 5 10 20]
            X::Matrix{Float64} = reference_mixtures * reference_signatures
            X[1, 1] = NaN
            result::Tuple = NMFk.execute(
                X,
                2:3,
                2;
                cutoff=0.5,
                load=false,
                save=false,
                mixture=:mixmatch,
                resultdir=result_directory,
                serial=true,
                seed=2015,
                maxiter=80,
                tol=1e-6,
                quiet=true
            )

            estimated_mixtures::Vector{Matrix{Float64}} = result[1]
            estimated_signatures::Vector{Matrix{Float64}} = result[2]
            fit::Vector{Float64} = result[3]
            robustness::Vector{Float64} = result[4]
            selected_rank::Union{Int, Nothing} = result[6]
            reconstructed::Matrix{Float64} = estimated_mixtures[2] * estimated_signatures[2]
            Test.@test count(isnan, X) == 1
            Test.@test selected_rank === 2
            Test.@test size(estimated_mixtures[2]) == (size(X, 1), 2)
            Test.@test size(estimated_signatures[2]) == (2, size(X, 2))
            Test.@test size(estimated_mixtures[3]) == (size(X, 1), 3)
            Test.@test size(estimated_signatures[3]) == (3, size(X, 2))
            Test.@test all(isapprox.(vec(sum(estimated_mixtures[2]; dims=2)), 1.0; atol=1e-8))
            Test.@test all(isfinite, estimated_mixtures[2])
            Test.@test all(isfinite, estimated_signatures[2])
            Test.@test isfinite(reconstructed[1, 1])
            Test.@test fit[2] < fit[3]
            Test.@test robustness[2] > robustness[3]
            Test.@test isempty(filter(name::String -> endswith(name, ".sha256"), readdir(result_directory)))
        end
    end

    Test.@testset "matrix execute forwards non-default method and clustering orientation" begin
        mktempdir() do result_directory::String
            X::Matrix{Float64} = [
                0.10 0.30 0.50 0.70
                0.20 0.40 0.60 0.80
                0.30 0.50 0.70 0.90
                0.40 0.60 0.80 0.95
                0.15 0.35 0.55 0.75
                0.25 0.45 0.65 0.85
            ]
            result::Tuple = NMFk.execute(
                X,
                1:1,
                1;
                load=false,
                save=false,
                method=:multmse,
                clusterWmatrix=true,
                resultdir=result_directory,
                serial=true,
                seed=29,
                maxiter=15,
                tol=1e-5,
                quiet=true
            )
            estimated_mixtures::Vector{Matrix{Float64}} = result[1]
            estimated_signatures::Vector{Matrix{Float64}} = result[2]
            selected_rank::Union{Int, Nothing} = result[6]
            Test.@test selected_rank === 1
            Test.@test isapprox(sum(estimated_mixtures[1]), 1.0; atol=1e-8)
            Test.@test !isapprox(sum(estimated_signatures[1]), 1.0; atol=1e-4)
            Test.@test isempty(filter(name::String -> endswith(name, ".sha256"), readdir(result_directory)))
        end
    end

    Test.@testset "execute(loadonly=true) missing file returns empties" begin
        mktempdir() do d::String
            X::Matrix{Float64} = ones(3, 3)
            W, H, fit, robustness, aic = NMFk.execute(X, 2, 1; loadonly=true, load=true, save=false, casefilename="case", resultdir=d, quiet=true)
            Test.@test size(W) == (0, 0)
            Test.@test size(H) == (0, 0)
            Test.@test fit == Inf
            Test.@test robustness == -1
            Test.@test aic == -Inf
            Test.@test !isfile(joinpath(d, "case_x_matrix_3_3.jld.sha256"))
        end
    end
end
