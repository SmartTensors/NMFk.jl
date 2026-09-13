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
				errorreader = @async read(errR, String);
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
    stdoutcaptureon()

Redirect `stdout` to an asynchronous in-memory reader.
Call [`stdoutcaptureoff`](@ref) to restore the original stream and collect the
captured text.

# Returns

The asynchronous `Task` that reads from the redirected stream.

# Examples

```julia
NMFk.stdoutcaptureon()
print("captured")
text = NMFk.stdoutcaptureoff()
@assert text == "captured"
```
"""
function stdoutcaptureon()
	global outputoriginal = stdout;
	(outR, outW) = redirect_stdout();
	global outputread = outR;
	global outputwrite = outW;
	global outputreader = @async read(outputread, String);
end

"""
    stdoutcaptureoff()

Restore the `stdout` stream saved by [`stdoutcaptureon`](@ref), close the
capture pipe, and return the captured text.

# Returns

The captured standard-output text as a `String`.
"""
function stdoutcaptureoff()
	redirect_stdout(outputoriginal);
	close(outputwrite);
	output = fetch(outputreader);
	close(outputread);
	return output
end

"""
    stderrcaptureon()

Redirect `stderr` to an asynchronous in-memory reader.
Call [`stderrcaptureoff`](@ref) to restore the original stream and collect the
captured text.

# Returns

The asynchronous `Task` that reads from the redirected stream.

# Examples

```julia
NMFk.stderrcaptureon()
print(stderr, "captured")
text = NMFk.stderrcaptureoff()
@assert text == "captured"
```
"""
function stderrcaptureon()
	global errororiginal = stderr;
	(errR, errW) = redirect_stderr();
	global errorread = errR;
	global errorwrite = errW;
	global errorreader = @async read(errorread, String);
end

"""
    stderrcaptureoff()

Restore the `stderr` stream saved by [`stderrcaptureon`](@ref), close the
capture pipe, and return the captured text.

# Returns

The captured standard-error text as a `String`.
"""
function stderrcaptureoff()
	redirect_stderr(errororiginal);
	close(errorwrite);
	erroro = fetch(errorreader)
	close(errorread);
	return erroro
end

"""
    stdouterrcaptureon()

Redirect both `stdout` and `stderr` to asynchronous in-memory readers.
Call [`stdouterrcaptureoff`](@ref) to restore both streams and collect their
captured text.

# Returns

The asynchronous `Task` that reads from the redirected `stderr` stream.
"""
function stdouterrcaptureon()
	stdoutcaptureon()
	stderrcaptureon()
end

"""
    stdouterrcaptureoff()

Restore the streams saved by [`stdouterrcaptureon`](@ref), close both capture
pipes, and return their captured text separately.

# Returns

A tuple `(stdout_text, stderr_text)` containing two `String` values.

# Examples

```julia
NMFk.stdouterrcaptureon()
print("output")
print(stderr, "error")
stdout_text, stderr_text = NMFk.stdouterrcaptureoff()
@assert (stdout_text, stderr_text) == ("output", "error")
```
"""
function stdouterrcaptureoff()
	return stdoutcaptureoff(), stderrcaptureoff()
end

"""
    quieton()

Set `NMFk.global_quiet` to `true`, making it the default `quiet` value for
NMFk operations that consult this module-level setting.

# Returns

`true`, the updated value of `NMFk.global_quiet`.

# Examples

```julia
NMFk.quieton()
@assert NMFk.global_quiet
```
"""
function quieton()
	global global_quiet = true;
end

"""
    quietoff()

Set `NMFk.global_quiet` to `false`, making it the default `quiet` value for
NMFk operations that consult this module-level setting.

# Returns

`false`, the updated value of `NMFk.global_quiet`.

# Examples

```julia
NMFk.quietoff()
@assert !NMFk.global_quiet
```
"""
function quietoff()
	global global_quiet = false;
end
