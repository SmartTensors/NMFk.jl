# Documentation authoring

NMFk keeps its user-facing documentation in ordinary source-attached Julia docstrings and uses Documenter.jl to publish them.
DocumentFunction.jl is a documentation-development dependency that scans source without loading NMFk, prepares conservative drafts, and gathers call-site evidence for human or Codex review.
It is not an NMFk runtime dependency, and the authoring command never edits source files automatically.

## Set up the authoring environment

Run the commands on this page from the repository root with Julia 1.12.

```bash
julia --startup-file=no --project=docs -e 'import Pkg; Pkg.instantiate()'
```

DocumentFunction 1.6 is resolved from the Julia General registry through `docs/Project.toml`.

## Audit the current docstrings

```bash
julia --startup-file=no --project=docs scripts/document.jl --check
```

This deterministic, offline check is also run before every CI documentation build.
It audits the documented and public-discoverable functions that DocumentFunction can identify in NMFk's legacy source layout.
Use the stricter parameter audit while improving individual docstrings, but do not treat its current findings as a release failure:

```bash
julia --startup-file=no --project=docs scripts/document.jl --check --require-parameters
```

## Collect evidence and prepare a draft

Collect signatures, implementation context, tests, examples, demonstrations, notebooks, and existing documentation for one function:

```bash
julia --startup-file=no --project=docs scripts/document.jl --prompt --function=NMFk.execute --output=docs/build/execute-context.md
```

The generated evidence file is under the ignored `docs/build` directory and can be given to Codex or reviewed directly.
Generate a conservative, non-mutating draft for the same function with:

```bash
julia --startup-file=no --project=docs scripts/document.jl --draft --function=NMFk.execute --output=docs/build/execute-draft.md
```

Use `--missing --all` to inspect undocumented internal functions or `--changed --all` to limit draft and prompt output to changed Julia source files.
Generated `TODO` text identifies missing semantic evidence and must be replaced by a reviewer before a docstring is added to source.

## Supply descriptions and verify an example

The library API accepts explicit descriptions for positional arguments, keywords, and return values.
It can also execute a completed example twice in fresh Julia processes to check that the example succeeds and produces deterministic output.
Precompile the NMFk project before the first isolated verification so one-time package setup messages are not treated as example diagnostics:

```bash
julia --startup-file=no --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The following targeted recipe uses the lightweight `NMFk.setdpi` function:

```julia
import DocumentFunction

repository::String = pwd()
source::String = joinpath(repository, "src")
specs::Vector{DocumentFunction.FunctionSpec} = DocumentFunction.scanfunctions(
	source;
	public_only=false,
	module_name="NMFk",
)
spec::DocumentFunction.FunctionSpec = only(filter(
	candidate::DocumentFunction.FunctionSpec -> candidate.qualified_name == "NMFk.setdpi",
	specs,
))
draft::DocumentFunction.DocumentationDraft = DocumentFunction.draftdocumentation(
	spec;
	summary="Set the module-wide image resolution used by NMFk plotting functions.",
	argtext=Dict{Symbol, String}(:dpi => "Image resolution to store in `NMFk.imagedpi`."),
	returntext="Return the configured image resolution.",
	example_values=Dict{Symbol, String}(:dpi => "150"),
)
verification::Tuple{
	DocumentFunction.DocumentationDraft,
	Vector{DocumentFunction.ExampleVerification},
} = DocumentFunction.verifyexamples(
	draft;
	project=repository,
	timeout_seconds=60,
)
all(result::DocumentFunction.ExampleVerification -> result.success, last(verification))
```

Add `keytext=Dict{Symbol, String}(:keyword => "Description.")` when the selected function has keywords.
Only copy reviewed, verified output into the source docstring; do not replace an existing docstring from an unreviewed draft.
