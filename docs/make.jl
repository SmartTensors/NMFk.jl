import Documenter
import NMFk

Documenter.makedocs(
	modules = [NMFk],
	authors = "Velimir V. Vesselinov and contributors",
	repo = Documenter.Remotes.GitHub("SmartTensors", "NMFk.jl"),
	sitename = "NMFk.jl",
	format = Documenter.HTML(
		canonical = "https://smarttensors.github.io/NMFk.jl/stable/",
		edit_link = "master",
		prettyurls = get(ENV, "CI", "false") == "true",
	),
	checkdocs = :none,
	pages = [
		"Home" => "index.md",
		"Getting started" => "getting-started.md",
		"Guides and research" => "guides.md",
		"API reference" => "api.md",
		"Documentation authoring" => "documentation-authoring.md",
	],
)

Documenter.deploydocs(
	repo = "github.com/SmartTensors/NMFk.jl.git",
	devbranch = "master",
	push_preview = false,
	versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
