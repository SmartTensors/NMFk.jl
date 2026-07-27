# Agent Guidance

## Scope

These instructions apply to the entire NMFk repository.

Follow more specific instructions if a nested `AGENTS.md` is added later.

## Julia style

- Use explicit package imports such as `import LinearAlgebra`.
- Do not use `using`.
- Add explicit types to variables, arguments, and return values.
- Avoid introducing `try`/`catch` statements.
- Prefer small functions with explicit inputs and outputs over implicit global state.
- Preserve existing public APIs unless an API change is explicitly requested.
- Follow the repository formatting rules in `.JuliaFormatter.toml`.
- Do not perform unrelated formatting or mechanical rewrites.

## Julia environment

Use Julia 1.11 unless a task explicitly targets another version.

Run Julia without user startup-file customizations:

```powershell
julia +1.11 --startup-file=no --project=.
```

NMFk depends on Mads.

Respect the sibling Mads development checkout when it is active.

Do not replace sibling development packages with registry versions merely to
make dependency resolution easier.

Preserve `Manifest-v1.12.toml` as a separate Julia 1.12 environment artifact.

## Repository layout

- `src/` contains the active NMFk implementation.
- `test/` contains unit, integration, and workflow tests.
- `docs/` contains documentation sources and tooling.
- `examples/`, `demo/`, and `notebooks/` contain user workflows.
- `src_old/` and `deps_old/` are legacy directories.
- `images/`, `movies/`, and workflow result directories may contain generated artifacts.

Do not edit legacy or generated artifacts unless the task explicitly places
them in scope.

## Testing

Start with the narrowest test that covers the change.

Important focused test files include:

- `test/test_information_theory.jl`
- `test/test_execute_smoke.jl`
- `test/test_execute_hash.jl`
- `test/test_normalize.jl`
- `test/test_checks.jl`
- `test/test_input_checks.jl`

For the complete NMFk test suite:

```powershell
julia +1.11 --startup-file=no --project=. -e 'import NMFk; NMFk.test()'
```

The standard package entry point is also valid:

```powershell
julia +1.11 --startup-file=no --project=. -e 'import Pkg; Pkg.test()'
```

Full imports, plotting backends, optimization solvers, and repeated NMF
factorizations can make the complete suite expensive.

Use small deterministic matrices or tensors for focused smoke tests.

When practical, set an explicit random seed in reproducibility tests.

## NMFk behavior

Preserve saved-result compatibility, matrix hashes, and `load=true` and
`save=true` behavior.

Changes to `execute` must validate:

- nonnegative input handling;
- normalization warnings;
- missing-value behavior;
- rank-range behavior;
- cache and input-hash correctness;
- returned `W`, `H`, fit, robustness, AIC, and selected-rank values.

Do not silently alter rank-selection or silhouette semantics.

Silhouette thresholds are analysis-specific.

Do not assume that 0.5 is the universal threshold for a useful solution.

Preserve the existing NMFk information-theory implementation while TensIT
compatibility is required.

Do not remove or redirect public information-theory functions without an
explicit migration request and compatibility plan.

## Documentation

Update documentation and examples when public behavior changes.

Keep examples consistent with the actual API and saved-result conventions.

Break Markdown prose into one sentence per line when practical so text changes
remain easy to review in Git.

## Compatibility and safety

Do not delete cached decompositions, saved matrices, hashes, figures, or user
results without explicit authorization.

Keep changes narrowly scoped and inspect the worktree before editing.

After changes, run:

```powershell
git diff --check
git status --short
```
