import Pkg
import DocumentFunction

function welcome()
	c = Base.text_colors
	tx = c[:normal] # text
	bl = c[:bold] # bold
	d1 = c[:bold] * c[:blue]    # first dot
	d2 = c[:bold] * c[:red]     # second dot
	d3 = c[:bold] * c[:green]   # third dot
	d4 = c[:bold] * c[:magenta] # fourth dot
	println("$(bl)NMFk: Nonnegative Matrix Factorization + k-means clustering and physics constraints$(tx)")
	println("====")
	println("")
	println("$(d1)  _     _  $(d2) _      _  $(d3) _______   $(d4)_$(tx)")
	println("$(d1) |  \\  | | $(d2)|  \\  /  | $(d3)|  _____| $(d4)| |  _$(tx)")
	println("$(d1) | . \\ | | $(d2)| . \\/ . | $(d3)| |___    $(d4)| | / /$(tx)")
	println("$(d1) | |\\ \\| | $(d2)| |\\  /| | $(d3)|  ___|   $(d4)| |/ /$(tx)")
	println("$(d1) | | \\ ' | $(d2)| | \\/ | | $(d3)| |       $(d4)|   ($(tx)")
	println("$(d1) | |  \\  | $(d2)| |    | | $(d3)| |       $(d4)| |\\ \\$(tx)")
	println("$(d1) |_|   \\_| $(d2)|_|    |_| $(d3)|_|       $(d4)|_| \\_\\$(tx)")
	println("")
	println("NMFk performs unsupervised machine learning based on matrix decomposition coupled with various constraints.")
	println("NMFk provides automatic identification of the optimal number of signals (features) present in two-dimensional data arrays (matrices).")
	println("NMFk offers visualization, pre-, and post-processing capabilities.")
end

function functions(re::Regex; stdout::Bool=false, quiet::Bool=false)
	n = 0
	for i in modules
		Core.eval(NMFk, :(@tryimport $(Symbol(i))))
		n += functions(Symbol(i), re; stdout=stdout, quiet=quiet)
	end
	n > 0 && string == "" && @info("Total number of functions: $n")
	return
end
function functions(string::AbstractString=""; stdout::Bool=false, quiet::Bool=false)
	n = 0
	for i in modules
		Core.eval(NMFk, :(@tryimport $(Symbol(i))))
		n += functions(Symbol(i), string; stdout=stdout, quiet=quiet)
	end
	n > 0 && string == "" && @info("Total number of functions: $n")
	return
end
function functions(m::Union{Symbol, Module}, re::Regex; stdout::Bool=false, quiet::Bool=false)
	n = 0
	try
		f = names(eval(m); all=true)
		functions = Vector{String}(undef, 0)
		for i = eachindex(f)
			functionname = "$(f[i])"
			if occursin("eval", functionname) || occursin("#", functionname) || occursin("__", functionname) || functionname == "$m"
				continue
			end
			if ismatch(re, functionname)
				push!(functions, functionname)
			end
		end
		if length(functions) > 0
			!quiet && @info("$(m) functions:")
			sort!(functions)
			n = length(functions)
			if stdout
				!quiet && Base.display(TextDisplay(STDOUT), functions)
			else
				!quiet && Base.display(functions)
			end
		end
	catch
		@warn("Module $m not defined!")
	end
	n > 0 && string == "" && @info("Number of functions in module $m: $n")
	return n
end
function functions(m::Union{Symbol, Module}, string::AbstractString=""; stdout::Bool=false, quiet::Bool=false)
	n = 0
	if string != ""
		quiet=false
	end
	try
		f = names(Core.eval(NMFk, m); all=true)
		functions = Vector{String}(undef, 0)
		for i = eachindex(f)
			functionname = "$(f[i])"
			if occursin("eval", functionname) || occursin("#", functionname) || occursin("__", functionname) || functionname == "$m"
				continue
			end
			if string == "" || occursin(string, functionname)
				push!(functions, functionname)
			end
		end
		if length(functions) > 0
			!quiet && @info("$(m) functions:")
			sort!(functions)
			n = length(functions)
			if stdout
				!quiet && Base.display(TextDisplay(STDOUT), functions)
			else
				!quiet && Base.display(functions)
			end
		end
	catch
		@warn("Module $m not defined!")
	end
	n > 0 && string == "" && @info("Number of functions in module $m: $n")
	return n
end
@doc """
List available functions in the NMFk modules:

$(DocumentFunction.documentfunction(functions;
argtext=Dict("string"=>"string to display functions with matching names",
			"m"=>"NMFk module")))

Examples:

```julia
NMFk.functions()
NMFk.functions("get")
NMFk.functions(NMFk, "get")
```
""" functions

"Checks if package is available"
function ispkgavailable(modulename::AbstractString)
	return pkginstalled(modulename)
end

function pkginstalled(modulename::AbstractString)
	found = false
	deps = Pkg.dependencies()
	for (uuid, dep) in deps
		dep.is_direct_dep || continue
		isnothing(dep.version) && continue
		if dep.name == modulename
			found = true
			break
		end
	end
	return found
end
function pkginstalled()
	deps = Pkg.dependencies()
	installs = Dict{String, VersionNumber}()
	for (uuid, dep) in deps
		dep.is_direct_dep || continue
		isnothing(dep.version) && continue
		installs[dep.name] = dep.version
	end
	return installs
end

"Print error message"
function printerrormsg(errmsg::Any)
	Base.showerror(Base.stderr, errmsg)
	try
		if in(:msg, fieldnames(errmsg))
			@warn(strip(errmsg.msg))
		elseif typeof(errmsg) <: AbstractString
			@warn(errmsg)
		end
	catch
		@warn(errmsg)
	end
end

"Try to import a module"
macro tryimport(s::Symbol, domains::Symbol=:NMFk)
	mname = string(s)
	domain = eval(domains)
	if !ispkgavailable(mname)
		try
			Pkg.add(mname)
		catch
			@info string("Module ", s, " is not available!")
			return nothing
		end
	end
	if !isdefined(domain, s)
		importq = string(:(import $s))
		warnstring = string("Module ", s, " cannot be imported!")
		q = quote
			try
				Core.eval($domain, Meta.parse($importq))
			catch errmsg
				printerrormsg(errmsg)
				@warn($warnstring)
			end
		end
		return :($(esc(q)))
	end
end