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

## Architecture and dependency boundaries

`src/NMFk.jl` is the composition root for matrix and tensor factorization, clustering, constrained solvers, preprocessing, persistence, and plotting.
Keep input validation and normalization in `NMFkChecks.jl` and `NMFkPreprocess.jl`, factorization and rank sweeps in `NMFkExecute.jl` and the matrix/tensor layers, and result ordering and selection in the cluster/finalize layers.
`NMFkIO.jl`, `NMFkRestart.jl`, and the hash helpers in `NMFkExecute.jl` jointly own saved-run compatibility; analysis code must not invent alternate filenames or bypass input hashes.
Preserve the orientation and meaning of `W`, `H`, fit, robustness, AIC, `kopt`, and the fields of `NMFkResult` and `NMFkSweepResult`.

Mads is a direct coordinated dependency and supplies shared utilities used by downstream workflows.
Do not copy Mads behavior into NMFk or replace a sibling development checkout with a registry package to resolve locally.
Keep any NMFk/TensIT information-theory compatibility surface explicit and additive; moving or removing an established function requires a migration plan and tests in both consumers.

## Test and artifact boundaries

The root suite groups lightweight utilities, execute and hash behavior, I/O, preprocessing, and larger optimization workflows.
Run the smallest corresponding file first, use fixed seeds, and reserve the full suite for changes that cross factorization, solver, persistence, or plotting boundaries.
Do not treat an unavailable optimizer, plotting backend, or expensive integration section as evidence that the affected behavior passed.

Saved `.jld` matrices and factors, `.sha256` sidecars, restart data, figures, movies, and notebook outputs are generated scientific artifacts.
Never delete or silently regenerate them to make a test pass, and never accept a hash mismatch without tracing the input change.
Keep the Julia 1.11 `Manifest.toml` and the separate Julia 1.12 manifest artifact distinct; dependency resolution must target the intended environment and its diff must be reviewed.
