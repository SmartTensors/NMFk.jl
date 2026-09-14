import DocumentFunction

const REPOSITORY_ROOT::String = normpath(joinpath(@__DIR__, ".."))

Base.@kwdef mutable struct Options
	mode::Symbol = :check
	source::String = "src"
	output::String = ""
	function_name::String = ""
	public_only::Bool = true
	changed_only::Bool = false
	require_parameters::Bool = false
	max_callsites::Int = 8
end

function _usage()::String
	return """
Usage:
  julia --startup-file=no --project=docs scripts/document.jl [mode] [options]

Modes:
  --check       Audit source documentation without changing files (default).
  --draft       Render conservative offline documentation drafts.
  --missing     Render drafts only for undocumented functions.
  --prompt      Emit source and call-site evidence for a reviewer.
  --baseline    Accept reviewed source hashes in .documentfunction.toml.

Options:
  --source=PATH             Source file or directory (default: src).
  --output=PATH             Write draft, missing, or prompt output to a file.
  --function=NAME           Select one function for draft, missing, or prompt output.
  --changed                 Limit draft or prompt output to changed Julia files.
  --all                     Include non-public and undocumented functions.
  --require-parameters      Require structured parameter descriptions in --check.
  --max-callsites=N         Maximum evidence call sites per function.
"""
end

function _validate_options(options::Options)::Nothing
	render_mode::Bool = options.mode in (:draft, :missing, :prompt)
	if !render_mode && (!isempty(options.output) || !isempty(options.function_name) || options.changed_only)
		throw(ArgumentError("--output, --function, and --changed require --draft, --missing, or --prompt"))
	end
	if options.require_parameters && options.mode != :check
		throw(ArgumentError("--require-parameters requires --check"))
	end
	return nothing
end

function _parse_options(arguments::Vector{String})::Options
	options::Options = Options()
	for argument::String in arguments
		if argument == "--check"
			options.mode = :check
		elseif argument == "--draft"
			options.mode = :draft
		elseif argument == "--missing"
			options.mode = :missing
		elseif argument == "--prompt"
			options.mode = :prompt
		elseif argument == "--baseline"
			options.mode = :baseline
		elseif argument == "--changed"
			options.changed_only = true
		elseif argument == "--all"
			options.public_only = false
		elseif argument == "--require-parameters"
			options.require_parameters = true
		elseif startswith(argument, "--source=")
			options.source = split(argument, '='; limit=2)[2]
		elseif startswith(argument, "--output=")
			options.output = split(argument, '='; limit=2)[2]
		elseif startswith(argument, "--function=")
			options.function_name = split(argument, '='; limit=2)[2]
		elseif startswith(argument, "--max-callsites=")
			options.max_callsites = parse(Int, split(argument, '='; limit=2)[2])
		elseif argument == "--help" || argument == "-h"
			options.mode = :help
		else
			throw(ArgumentError("unknown argument: $(argument)"))
		end
	end
	if options.max_callsites < 1
		throw(ArgumentError("--max-callsites must be positive"))
	end
	_validate_options(options)
	return options
end

function _repository_path(path::AbstractString)::String
	return isabspath(path) ? normpath(String(path)) : normpath(joinpath(REPOSITORY_ROOT, path))
end

function _changed_julia_files()::Vector{String}
	tracked_text::String = read(`git -C $REPOSITORY_ROOT diff --name-only --diff-filter=ACMR HEAD -- '*.jl'`, String)
	untracked_text::String = read(`git -C $REPOSITORY_ROOT ls-files --others --exclude-standard -- '*.jl'`, String)
	paths::Vector{String} = String[]
	for relative_path::SubString{String} in split("$(tracked_text)\n$(untracked_text)", '\n')
		stripped_path::String = strip(String(relative_path))
		candidate::String = joinpath(REPOSITORY_ROOT, stripped_path)
		if !isempty(stripped_path) && isfile(candidate)
			push!(paths, abspath(candidate))
		end
	end
	return unique(sort(paths))
end

function _named_specs(
	options::Options,
	specs::Vector{DocumentFunction.FunctionSpec},
)::Vector{DocumentFunction.FunctionSpec}
	if isempty(options.function_name)
		return specs
	end
	selected::Vector{DocumentFunction.FunctionSpec} = filter(specs) do spec::DocumentFunction.FunctionSpec
		short_name::String = last(split(spec.qualified_name, '.'))
		return spec.qualified_name == options.function_name || short_name == options.function_name
	end
	if isempty(selected)
		throw(ArgumentError("no scanned function matches --function=$(options.function_name)"))
	end
	return selected
end

function _selected_specs(options::Options)::Vector{DocumentFunction.FunctionSpec}
	specs::Vector{DocumentFunction.FunctionSpec} = DocumentFunction.scanfunctions(
		_repository_path(options.source);
		public_only=options.public_only,
		module_name="NMFk",
	)
	if !options.changed_only
		return _named_specs(options, specs)
	end
	changed_files::Set{String} = Set(_changed_julia_files())
	changed_specs::Vector{DocumentFunction.FunctionSpec} = filter(specs) do spec::DocumentFunction.FunctionSpec
		return any(method::DocumentFunction.MethodSpec -> abspath(method.source.path) in changed_files, spec.methods)
	end
	return _named_specs(options, changed_specs)
end

function _evidence_paths()::Vector{String}
	paths::Vector{String} = [
		joinpath(REPOSITORY_ROOT, "test"),
		joinpath(REPOSITORY_ROOT, "examples"),
		joinpath(REPOSITORY_ROOT, "demo"),
		joinpath(REPOSITORY_ROOT, "notebooks"),
		joinpath(REPOSITORY_ROOT, "docs"),
	]
	return filter(isdir, paths)
end

function _render_specs(options::Options, specs::Vector{DocumentFunction.FunctionSpec})::String
	selected::Vector{DocumentFunction.FunctionSpec} = if options.mode == :missing
		filter(spec::DocumentFunction.FunctionSpec -> isnothing(spec.existing_doc), specs)
	else
		specs
	end
	blocks::Vector{String} = String[]
	for spec::DocumentFunction.FunctionSpec in selected
		if options.mode == :prompt
			push!(blocks, DocumentFunction.documentationcontext(
				spec;
				root=REPOSITORY_ROOT,
				evidence_paths=_evidence_paths(),
				max_callsites=options.max_callsites,
			))
		else
			draft::DocumentFunction.DocumentationDraft = DocumentFunction.draftdocumentation(spec)
			push!(blocks, "# $(spec.qualified_name)\n\n$(DocumentFunction.renderdocstring(draft))")
		end
	end
	return join(blocks, "\n\n---\n\n")
end

function _write_output(rendered::AbstractString, output::AbstractString)::Nothing
	if isempty(output)
		println(rendered)
		return nothing
	end
	output_path::String = _repository_path(output)
	mkpath(dirname(output_path))
	write(output_path, String(rendered))
	println("Wrote $(output_path)")
	return nothing
end

function main(arguments::Vector{String})::Int
	options::Options = _parse_options(arguments)
	if options.mode == :help
		println(_usage())
		return 0
	end
	source::String = _repository_path(options.source)
	if options.mode == :check
		issues::Vector{DocumentFunction.DocumentationIssue} = DocumentFunction.checkdocs(
			source;
			public_only=options.public_only,
			require_parameters=options.require_parameters,
			module_name="NMFk",
		)
		if isempty(issues)
			println("Documentation checks passed.")
			return 0
		end
		println(DocumentFunction.formatissues(issues; root=REPOSITORY_ROOT))
		return 1
	end
	if options.mode == :baseline
		path::String = DocumentFunction.writeinventory(
			source;
			public_only=options.public_only,
			module_name="NMFk",
		)
		println("Wrote $(path)")
		return 0
	end
	specs::Vector{DocumentFunction.FunctionSpec} = _selected_specs(options)
	_write_output(_render_specs(options, specs), options.output)
	return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
	exit(main(ARGS))
end
