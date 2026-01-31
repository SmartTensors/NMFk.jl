import NMFk
import Test

Test.@testset "capture + io utilities" begin
	Test.@testset "stdout/stderr capture toggles" begin
		# stdout
		out = ""
		NMFk.stdoutcaptureon()
		try
			println("hello-stdout")
		finally
			out = NMFk.stdoutcaptureoff()
		end
		Test.@test occursin("hello-stdout", out)

		# stderr
		err = ""
		NMFk.stderrcaptureon()
		try
			println(stderr, "hello-stderr")
		finally
			err = NMFk.stderrcaptureoff()
		end
		Test.@test occursin("hello-stderr", err)

		# both
		out2 = ""
		err2 = ""
		NMFk.stdouterrcaptureon()
		try
			println("hello-stdout-2")
			println(stderr, "hello-stderr-2")
		finally
			out2, err2 = NMFk.stdouterrcaptureoff()
		end
		Test.@test occursin("hello-stdout-2", out2)
		Test.@test occursin("hello-stderr-2", err2)
	end

	Test.@testset "joinpathcheck" begin
		root = mktempdir()
		p = NMFk.joinpathcheck(root, "a", "b", "file.txt")
		Test.@test ispath(dirname(p))
		Test.@test isdir(joinpath(root, "a", "b"))
	end
end
