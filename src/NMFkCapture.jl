import DocumentFunction

"""
Capture stdout of a block
"""
macro stdoutcapture(block)
	if quiet
		quote
			if ccall(:jl_generating_output, Cint, ()) == 0
				outputoriginal = stdout;
				(outR, outW) = redirect_stdout();
				outputreader = @async read(outR, String);
				evalvalue = $(esc(block))
				redirect_stdout(outputoriginal);
				close(outW);
				close(outR);
				return evalvalue
			end
		end
	else
		quote
			evalvalue = $(esc(block))
		end
	end
end

"""
Capture stderr of a block
"""
macro stderrcapture(block)
	if quiet
		quote
			if ccall(:jl_generating_output, Cint, ()) == 0
				errororiginal = stderr;
				(errR, errW) = redirect_stderr();
				errorreader = @async read(errR, String);
				evalvalue = $(esc(block))
				redirect_stderr(errororiginal);
				close(errW);
				close(errR);
				return evalvalue
			end
		end
	else
		quote
			evalvalue = $(esc(block))
		end
	end
end

"""
Capture stderr & stderr of a block
"""
macro stdouterrcapture(block)
	if quiet
		quote
			if ccall(:jl_generating_output, Cint, ()) == 0
				outputoriginal = stdout;
				(outR, outW) = redirect_stdout();
				outputreader = @async read(outR, String);
				errororiginal = stderr;
				(errR, errW) = redirect_stderr();
				errorreader = @async read(errR), String;
				evalvalue = $(esc(block))
				redirect_stdout(outputoriginal);
				close(outW);
				close(outR);
				redirect_stderr(errororiginal);
				close(errW);
				close(errR);
				return evalvalue
			end
		end
	else
		quote
			evalvalue = $(esc(block))
		end
	end
end

"""
Redirect stdout to a reader

$(DocumentFunction.documentfunction(stdoutcaptureon))
"""
function stdoutcaptureon()
	global outputoriginal = stdout;
	(outR, outW) = redirect_stdout();
	global outputread = outR;
	global outputwrite = outW;
	global outputreader = @async read(outputread, String);
end

"""
Restore stdout

$(DocumentFunction.documentfunction(stdoutcaptureoff))

Returns:

- standered output
"""
function stdoutcaptureoff()
	redirect_stdout(outputoriginal);
	close(outputwrite);
	output = wait(outputreader);
	close(outputread);
	return output
end

"""
Redirect stderr to a reader

$(DocumentFunction.documentfunction(stderrcaptureon))
"""
function stderrcaptureon()
	global errororiginal = stderr;
	(errR, errW) = redirect_stderr();
	global errorread = errR;
	global errorwrite = errW;
	global errorreader = @async read(errorread, String);
end

"""
Restore stderr

$(DocumentFunction.documentfunction(stderrcaptureoff))

Returns:

- standered error
"""
function stderrcaptureoff()
	redirect_stderr(errororiginal);
	close(errorwrite);
	erroro = wait(errorreader)
	close(errorread);
	return erroro
end

"""
Redirect stdout & stderr to readers

$(DocumentFunction.documentfunction(stdouterrcaptureon))
"""
function stdouterrcaptureon()
	stdoutcaptureon()
	stderrcaptureon()
end

"""
Restore stdout & stderr

$(DocumentFunction.documentfunction(stdouterrcaptureoff))

Returns:

- standered output and standered error
"""
function stdouterrcaptureoff()
	return stdoutcaptureoff(), stderrcaptureoff()
end

"""
Make NMFk quiet

$(DocumentFunction.documentfunction(quieton))
"""
function quieton()
	global quiet = true;
end

"""
Make NMFk not quiet

$(DocumentFunction.documentfunction(quietoff))
"""
function quietoff()
	global quiet = false;
end
