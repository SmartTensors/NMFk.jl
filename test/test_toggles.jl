import Test
import NMFk

Test.@testset "Global state toggles" begin
	Test.@testset "quieton/quietoff" begin
		orig = NMFk.global_quiet
		try
			NMFk.quietoff()
			Test.@test NMFk.global_quiet == false
			NMFk.quieton()
			Test.@test NMFk.global_quiet == true
		finally
			orig ? NMFk.quieton() : NMFk.quietoff()
		end
	end

	Test.@testset "restarton/restartoff" begin
		orig = NMFk.restart
		mktempdir() do d
			cd(d) do
				try
					NMFk.restartoff()
					Test.@test NMFk.restart == false
					NMFk.restarton()
					Test.@test NMFk.restart == true
					Test.@test isdefined(NMFk, :execute_singlerun_r3)
				finally
					NMFk.restartoff()
					orig && NMFk.restarton()
				end
			end
		end
	end
end
