import Test
import NMFk
import Distributed
import LinearAlgebra
import Random

Test.@testset "Seeded parallel matrix execute and cache" begin
    worker_ids::Vector{Int} = Distributed.addprocs(2; exeflags=Cmd(["--startup-file=no", "--project=@v1.12", "--threads=1", "--gcthreads=1"]))
    Distributed.remotecall_eval(Main, worker_ids, quote
        import NMFk
        import LinearAlgebra
        LinearAlgebra.BLAS.set_num_threads(1)
    end)
    LinearAlgebra.BLAS.set_num_threads(1)
    rng::Random.MersenneTwister = Random.MersenneTwister(42)
    matrix::Matrix{Float64} = Random.rand(rng, 10, 5)
    matrix[2, 3] = NaN
    mktempdir() do directory::String
        execution::Tuple = NMFk.execute(copy(matrix), 2:3, 3; seed=42, maxiter=40, serial=false,
            resultdir=directory, casefilename="parallel", load=false, save=true, quiet=true)
        Test.@test size(execution[1][2]) == (10, 2)
        Test.@test size(execution[2][3]) == (3, 5)
        Test.@test all(isfinite, execution[1][3])
        Test.@test all(isfinite, execution[3][2:3])
        Test.@test all(isfinite, execution[4][2:3])
        Test.@test all(isfinite, execution[5][2:3])
        loaded::Tuple = NMFk.execute(copy(matrix), 2, 3; resultdir=directory, casefilename="parallel",
            load=true, save=false, quiet=true)
        Test.@test loaded[1] == execution[1][2]
        Test.@test loaded[2] == execution[2][2]
    end
    Distributed.rmprocs(worker_ids)
end
