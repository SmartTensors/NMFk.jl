import Test
import NMFk

Test.@testset "Execute: cache hashing helpers" begin

	Test.@testset "hash_sha256_hex" begin
		X = Float64[1 2; 3 4]
		h1 = NMFk.hash_sha256_hex(X)
		Test.@test length(h1) == 64
		Test.@test all(c -> (c in '0':'9') || (c in 'a':'f'), collect(h1))

		h2 = NMFk.hash_sha256_hex(copy(X))
		Test.@test h1 == h2

		X2 = Float64[1 2; 3 5]
		Test.@test NMFk.hash_sha256_hex(X2) != h1
	end

	Test.@testset "check_x_hash! writes/validates .sha256" begin
		X = Float64[1 2; 3 4]
		tmp = mktempdir()
		xfile = joinpath(tmp, "nmfk_x_matrix_2_2.jld")
		h = NMFk.check_x_hash!(X, xfile)
		hashfile = xfile * ".sha256"
		Test.@test isfile(hashfile)
		stored = strip(read(hashfile, String))
		Test.@test stored == h

		# Second call should not error and should return the same hash
		h2 = NMFk.check_x_hash!(X, xfile)
		Test.@test h2 == h

		# If the stored hash is wrong, we warn but should still return computed hash
		open(hashfile, "w") do io
			write(io, "deadbeef")
			write(io, "\n")
		end
		h3 = NMFk.check_x_hash!(X, xfile)
		Test.@test h3 == h

		read_only_xfile::String = joinpath(tmp, "read_only_matrix.jld")
		read_only_hash::String =
			NMFk.check_x_hash!(X, read_only_xfile; write_missing=false)
		Test.@test read_only_hash == h
		Test.@test !isfile(read_only_xfile * ".sha256")
	end

end
