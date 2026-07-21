import Cairo
import Colors
import Compose
import Fontconfig
import Printf

# Run with:
# julia +1.11 --startup-file=no --project=C:\Users\monty\.julia\dev\NMFk docs\generate_tensor_information_research_summary_figures.jl
#
# The plotted policy values are the audited paper snapshot from the
# 2005–2023 binning-optimization JLD2 checkpoint generated on 20 July 2026.

const OUTPUT_DIRECTORY::String =
	joinpath(@__DIR__, "figures", "tensor_information_research_paper")
const FRAMEWORK_BASENAME::String = "tensor_information_framework_summary"
const POLICY_BASENAME::String = "oklahoma_policy_summary"
const FONT_REGULAR::String = "Arial"
const FONT_BOLD::String = "Arial Bold"

const RETENTION_MATRIX::Matrix{Float64} = [
	0.9029903 0.9492392 0.7321785 0.2030559 0.1792569 0.7565641
	0.9318670 0.9624911 0.8117976 0.4181353 0.4076655 0.6977449
	0.9513410 0.9727569 0.8625887 0.4443191 0.4337657 0.7565641
	0.9551001 0.9749422 0.8729948 0.4618683 0.4501808 0.7565641
]
const BALANCED_SCORES::Vector{Float64} = [
	0.398189,
	0.905313,
	0.962004,
	1.0,
]
const POSSIBLE_CELL_COSTS::Vector{Float64} = [
	1_062_945.0,
	56_837_100.0,
	201_687_000.0,
	287_916_369.0,
]
const ROW_LABELS::Vector{NTuple{2,String}} = [
	("2005–2023 · Event ≥90%", "15×13 / day"),
	("2005–2023 · Balanced ≥90%", "150×139 / 2 days"),
	("2005–2023 · Balanced ≥95%", "200×185 / day"),
	("2005–2023 · Finest candidate", "239×221 / day"),
]
const METRIC_LABELS::Vector{String} = [
	"Event",
	"Energy",
	"Magnitude",
	"Longitude",
	"Latitude",
	"Time",
]
const METRIC_HEADER_LINES::Vector{Vector{String}} = [
	["Event"],
	["Energy", "location"],
	["Magnitude", "location"],
	["Longitude"],
	["Latitude"],
	["Time"],
]
const POLICY_COLORS::Vector{String} = [
	"#6b7280",
	"#1f78b4",
	"#ff7f00",
	"#111827",
]

function text_context(
	x::Float64,
	y::Float64,
	label::String;
	fontsize_points::Float64 = 14.0,
	color::String = "#111827",
	font_name::String = FONT_REGULAR,
	horizontal_alignment::Any = Compose.hcenter,
	vertical_alignment::Any = Compose.vcenter,
)::Compose.Context
	return Compose.compose(
		Compose.context(),
		Compose.text(
			x,
			y,
			label,
			horizontal_alignment,
			vertical_alignment,
		),
		Compose.fill(color),
		Compose.font(font_name),
		Compose.fontsize(fontsize_points * Compose.pt),
	)
end

function multiline_context(
	x::Float64,
	y::Float64,
	lines::Vector{String};
	fontsize_points::Float64 = 14.0,
	line_spacing::Float64 = 0.032,
	color::String = "#111827",
	font_name::String = FONT_REGULAR,
	horizontal_alignment::Any = Compose.hcenter,
)::Compose.Context
	elements::Vector{Any} = Any[]
	line_count::Int = length(lines)
	first_y::Float64 = y - 0.5 * Float64(line_count - 1) * line_spacing
	for line_index::Int in eachindex(lines)
		push!(
			elements,
			text_context(
				x,
				first_y + Float64(line_index - 1) * line_spacing,
				lines[line_index];
				fontsize_points = fontsize_points,
				color = color,
				font_name = font_name,
				horizontal_alignment = horizontal_alignment,
			),
		)
	end
	return Compose.compose(Compose.context(), reverse(elements)...)
end

function box_context(
	x::Float64,
	y::Float64,
	width::Float64,
	height::Float64;
	fill_color::String,
	stroke_color::String,
	linewidth_millimeters::Float64 = 0.65,
)::Compose.Context
	return Compose.compose(
		Compose.context(),
		Compose.rectangle(x, y, width, height),
		Compose.fill(fill_color),
		Compose.stroke(stroke_color),
		Compose.linewidth(linewidth_millimeters * Compose.mm),
	)
end

function path_context(
	points::Vector{Tuple{Float64,Float64}};
	color::String,
	dashed::Bool = false,
	arrow_at_end::Bool = true,
	linewidth_millimeters::Float64 = 0.7,
)::Compose.Context
	if dashed
		return Compose.compose(
			Compose.context(),
			Compose.line(points),
			Compose.stroke(color),
			Compose.linewidth(linewidth_millimeters * Compose.mm),
			Compose.strokedash([1.5Compose.mm, 1.2Compose.mm]),
			Compose.arrow(arrow_at_end),
		)
	end
	return Compose.compose(
		Compose.context(),
		Compose.line(points),
		Compose.stroke(color),
		Compose.linewidth(linewidth_millimeters * Compose.mm),
		Compose.arrow(arrow_at_end),
	)
end

function circle_context(
	x::Float64,
	y::Float64,
	radius::Float64;
	fill_color::String,
	stroke_color::String = "#ffffff",
)::Compose.Context
	return Compose.compose(
		Compose.context(),
		Compose.circle(x, y, radius),
		Compose.fill(fill_color),
		Compose.stroke(stroke_color),
		Compose.linewidth(0.45Compose.mm),
	)
end

function retention_color(value::Float64)::Colors.RGB{Float64}
	clamped_value::Float64 = clamp(value, 0.0, 1.0)
	low::NTuple{3,Float64} = (68.0 / 255.0, 1.0 / 255.0, 84.0 / 255.0)
	middle::NTuple{3,Float64} = (33.0 / 255.0, 145.0 / 255.0, 140.0 / 255.0)
	high::NTuple{3,Float64} = (253.0 / 255.0, 231.0 / 255.0, 37.0 / 255.0)
	local first_color::NTuple{3,Float64}
	local second_color::NTuple{3,Float64}
	local weight::Float64
	if clamped_value <= 0.5
		first_color = low
		second_color = middle
		weight = 2.0 * clamped_value
	else
		first_color = middle
		second_color = high
		weight = 2.0 * clamped_value - 1.0
	end
	red::Float64 = (1.0 - weight) * first_color[1] + weight * second_color[1]
	green::Float64 = (1.0 - weight) * first_color[2] + weight * second_color[2]
	blue::Float64 = (1.0 - weight) * first_color[3] + weight * second_color[3]
	return Colors.RGB{Float64}(red, green, blue)
end

function text_color(background::Colors.RGB{Float64})::String
	luminance::Float64 =
		0.2126 * Float64(background.r) +
		0.7152 * Float64(background.g) +
		0.0722 * Float64(background.b)
	return luminance < 0.54 ? "#ffffff" : "#111827"
end

function percentage_label(value::Float64)::String
	return Printf.@sprintf("%.1f%%", 100.0 * value)
end

function abbreviated_cost(value::Float64)::String
	if value < 1.0e6
		return Printf.@sprintf("%.3fM", value / 1.0e6)
	elseif value < 10.0e6
		return Printf.@sprintf("%.2fM", value / 1.0e6)
	elseif value < 100.0e6
		return Printf.@sprintf("%.1fM", value / 1.0e6)
	end
	return Printf.@sprintf("%.0fM", value / 1.0e6)
end

function log_cost_position(
	value::Float64,
	x_minimum::Float64,
	x_maximum::Float64,
)::Float64
	log_minimum::Float64 = log10(5.0e5)
	log_maximum::Float64 = log10(4.0e8)
	fraction::Float64 =
		(log10(value) - log_minimum) / (log_maximum - log_minimum)
	return x_minimum + clamp(fraction, 0.0, 1.0) * (x_maximum - x_minimum)
end

function framework_figure()::Compose.Context
	elements::Vector{Any} = Any[]
	push!(
		elements,
		box_context(
			0.0,
			0.0,
			1.0,
			1.0;
			fill_color = "#ffffff",
			stroke_color = "#ffffff",
			linewidth_millimeters = 0.0,
		),
	)
	push!(
		elements,
		text_context(
			0.5,
			0.05,
			"Universal workflow for information-preserving tensor discretization";
			fontsize_points = 22.0,
			font_name = FONT_BOLD,
		),
	)
	push!(
		elements,
		text_context(
			0.5,
			0.095,
			"Raw information loss, marked-data loss, and tensor structure answer different questions";
			fontsize_points = 14.0,
			color = "#4b5563",
		),
	)

	push!(
		elements,
		box_context(
			0.01,
			0.34,
			0.14,
			0.32;
			fill_color = "#eff6ff",
			stroke_color = "#2563eb",
		),
	)
	push!(
		elements,
		multiline_context(
			0.08,
			0.385,
			["Raw", "observations"];
			fontsize_points = 12.5,
			line_spacing = 0.034,
			font_name = FONT_BOLD,
			color = "#1e3a8a",
		),
	)
	push!(
		elements,
		multiline_context(
			0.08,
			0.515,
			[
				"coordinates • time",
				"marks • weights",
				"explicit precision",
				"and origin",
			];
			fontsize_points = 11.0,
			line_spacing = 0.047,
		),
	)

	push!(
		elements,
		box_context(
			0.16,
			0.34,
			0.15,
			0.32;
			fill_color = "#f0fdf4",
			stroke_color = "#16a34a",
		),
	)
	push!(
		elements,
		multiline_context(
			0.235,
			0.385,
			["Candidate grids", "G_s"];
			fontsize_points = 12.5,
			line_spacing = 0.034,
			font_name = FONT_BOLD,
			color = "#14532d",
		),
	)
	push!(
		elements,
		multiline_context(
			0.235,
			0.515,
			[
				"longitude × latitude",
				"× time",
				"counts / weighted",
				"sums",
				"per-cell mark",
				"summaries",
				"possible-cell cost C_s",
			];
			fontsize_points = 10.0,
			line_spacing = 0.031,
		),
	)

	branch_x::Float64 = 0.34
	branch_width::Float64 = 0.25
	branch_height::Float64 = 0.20
	branch_centers::Vector{Float64} = [0.24, 0.50, 0.76]
	branch_y_values::Vector{Float64} =
		[center - 0.5 * branch_height for center::Float64 in branch_centers]
	branch_fills::Vector{String} = ["#eef2ff", "#fff7ed", "#f5f3ff"]
	branch_strokes::Vector{String} = ["#4f46e5", "#ea580c", "#7c3aed"]
	branch_titles::Vector{String} = [
		"Raw-to-grid retention",
		"Marked-data analysis",
		"Gridded-tensor structure",
	]
	branch_lines::Vector{Vector{String}} = [
		[
			"event • energy localization",
			"longitude • latitude • time",
			"I(X;G) / H(X)  and  H(X | G)",
		],
		[
			"magnitude-location dependence",
			"H(M | G) • MAE • RMSE",
			"reconstruction residual bits",
		],
		[
			"quantized value and lag MI",
			"coherence • residual coding",
			"spectral entropy / effective rank",
		],
	]
	for branch_index::Int in 1:3
		branch_y::Float64 = branch_y_values[branch_index]
		branch_center_y::Float64 = branch_centers[branch_index]
		push!(
			elements,
			box_context(
				branch_x,
				branch_y,
				branch_width,
				branch_height;
				fill_color = branch_fills[branch_index],
				stroke_color = branch_strokes[branch_index],
			),
		)
		push!(
			elements,
			text_context(
				branch_x + 0.5 * branch_width,
				branch_y + 0.038,
				branch_titles[branch_index];
				fontsize_points = 13.5,
				font_name = FONT_BOLD,
				color = branch_strokes[branch_index],
			),
		)
		push!(
			elements,
			multiline_context(
				branch_x + 0.5 * branch_width,
				branch_center_y + 0.025,
				branch_lines[branch_index];
				fontsize_points = 10.8,
				line_spacing = 0.038,
			),
		)
	end

	push!(
		elements,
		box_context(
			0.62,
			0.285,
			0.18,
			0.43;
			fill_color = "#fffbeb",
			stroke_color = "#d97706",
			linewidth_millimeters = 0.9,
		),
	)
	push!(
		elements,
		text_context(
			0.71,
			0.33,
			"Cross-grid selector";
			fontsize_points = 13.5,
			font_name = FONT_BOLD,
			color = "#92400e",
		),
	)
	push!(
		elements,
		multiline_context(
			0.71,
			0.515,
			[
				"six comparable metrics",
				"event • energy • magnitude",
				"longitude • latitude • time",
				"B_s = minimum",
				"relative retention",
				"Pareto • target • budget",
			];
			fontsize_points = 10.0,
			line_spacing = 0.039,
		),
	)

	push!(
		elements,
		box_context(
			0.83,
			0.35,
			0.16,
			0.30;
			fill_color = "#ecfdf5",
			stroke_color = "#059669",
			linewidth_millimeters = 0.9,
		),
	)
	push!(
		elements,
		text_context(
			0.91,
			0.395,
			"Auditable output";
			fontsize_points = 13.0,
			font_name = FONT_BOLD,
			color = "#065f46",
		),
	)
	push!(
		elements,
		multiline_context(
			0.91,
			0.525,
			[
				"trade-off family",
				"economical or",
				"high-fidelity policy",
				"not a universal",
				"optimum",
			];
			fontsize_points = 10.0,
			line_spacing = 0.040,
		),
	)

	push!(
		elements,
		path_context(
			[(0.15, 0.50), (0.16, 0.50)];
			color = "#2563eb",
		),
	)
	for target_y::Float64 in branch_centers
		push!(
			elements,
			path_context(
				[(0.31, 0.50), (0.325, 0.50), (0.325, target_y), (0.34, target_y)];
				color = "#64748b",
			),
		)
	end
	push!(
		elements,
		path_context(
			[(0.59, 0.24), (0.605, 0.24), (0.605, 0.40), (0.62, 0.40)];
			color = "#4f46e5",
		),
	)
	push!(
		elements,
		path_context(
			[(0.59, 0.50), (0.62, 0.50)];
			color = "#ea580c",
		),
	)
	push!(
		elements,
		path_context(
			[(0.80, 0.50), (0.83, 0.50)];
			color = "#059669",
		),
	)
	push!(
		elements,
		path_context(
			[(0.59, 0.76), (0.815, 0.76), (0.815, 0.62), (0.83, 0.62)];
			color = "#7c3aed",
			dashed = true,
		),
	)

	push!(
		elements,
		path_context(
			[(0.34, 0.925), (0.42, 0.925)];
			color = "#334155",
			arrow_at_end = false,
		),
	)
	push!(
		elements,
		text_context(
			0.43,
			0.925,
			"enters current cross-grid selection";
			fontsize_points = 12.5,
			horizontal_alignment = Compose.hleft,
		),
	)
	push!(
		elements,
		path_context(
			[(0.34, 0.965), (0.42, 0.965)];
			color = "#7c3aed",
			dashed = true,
			arrow_at_end = false,
		),
	)
	push!(
		elements,
		text_context(
			0.43,
			0.965,
			"reported diagnostic; not used for cross-grid ranking";
			fontsize_points = 12.5,
			horizontal_alignment = Compose.hleft,
		),
	)
	return Compose.compose(Compose.context(), reverse(elements)...)
end

function policy_summary_figure()::Compose.Context
	elements::Vector{Any} = Any[]
	push!(
		elements,
		box_context(
			0.0,
			0.0,
			1.0,
			1.0;
			fill_color = "#ffffff",
			stroke_color = "#ffffff",
			linewidth_millimeters = 0.0,
		),
	)
	push!(
		elements,
		text_context(
			0.5,
			0.04,
			"2005–2023 Oklahoma decision summary";
			fontsize_points = 22.0,
			font_name = FONT_BOLD,
		),
	)
	push!(
		elements,
		text_context(
			0.5,
			0.078,
			"Absolute retained-information fractions use one fixed 0–100% scale; cost is unnormalized";
			fontsize_points = 14.0,
			color = "#4b5563",
		),
	)

	absolute_x_minimum::Float64 = 0.27
	absolute_width::Float64 = 0.45
	metric_width::Float64 = absolute_width / Float64(length(METRIC_LABELS))
	balanced_x::Float64 = 0.735
	balanced_width::Float64 = 0.072
	cost_x_minimum::Float64 = 0.835
	cost_x_maximum::Float64 = 0.982
	rows_y_minimum::Float64 = 0.16
	row_height::Float64 = 0.145
	row_count::Int = size(RETENTION_MATRIX, 1)

	for metric_index::Int in eachindex(METRIC_LABELS)
		header_x::Float64 =
			absolute_x_minimum + (Float64(metric_index) - 0.5) * metric_width
		push!(
			elements,
			multiline_context(
				header_x,
				0.123,
				METRIC_HEADER_LINES[metric_index];
				fontsize_points = 11.0,
				line_spacing = 0.024,
				font_name = FONT_BOLD,
				color = "#374151",
			),
		)
	end
	push!(
		elements,
		multiline_context(
			balanced_x + 0.5 * balanced_width,
			0.124,
			["Balanced", "score*"];
			fontsize_points = 13.5,
			line_spacing = 0.025,
			font_name = FONT_BOLD,
			color = "#4338ca",
		),
	)
	push!(
		elements,
		multiline_context(
			0.5 * (cost_x_minimum + cost_x_maximum),
			0.124,
			["Possible grid cells", "(log scale)"];
			fontsize_points = 13.5,
			line_spacing = 0.025,
			font_name = FONT_BOLD,
			color = "#374151",
		),
	)

	cost_tick_values::Vector{Float64} = [1.0e6, 1.0e7, 1.0e8]
	cost_tick_labels::Vector{String} = ["1M", "10M", "100M"]
	for tick_index::Int in eachindex(cost_tick_values)
		tick_x::Float64 = log_cost_position(
			cost_tick_values[tick_index],
			cost_x_minimum,
			cost_x_maximum,
		)
		push!(
			elements,
			path_context(
				[
					(tick_x, rows_y_minimum),
					(tick_x, rows_y_minimum + Float64(row_count) * row_height),
				];
				color = "#d1d5db",
				arrow_at_end = false,
				linewidth_millimeters = 0.28,
			),
		)
		push!(
			elements,
			text_context(
				tick_x,
				rows_y_minimum + Float64(row_count) * row_height + 0.027,
				cost_tick_labels[tick_index];
				fontsize_points = 11.5,
				color = "#6b7280",
			),
		)
	end

	for row_index::Int in 1:row_count
		row_y::Float64 = rows_y_minimum + Float64(row_index - 1) * row_height
		row_center_y::Float64 = row_y + 0.5 * row_height
		row_background::String = isodd(row_index) ? "#f9fafb" : "#ffffff"
		push!(
			elements,
			box_context(
				0.01,
				row_y,
				0.98,
				row_height;
				fill_color = row_background,
				stroke_color = "#e5e7eb",
				linewidth_millimeters = 0.18,
			),
		)
		push!(
			elements,
			text_context(
				0.26,
				row_center_y - 0.012,
				ROW_LABELS[row_index][1];
				fontsize_points = 13.0,
				font_name = FONT_BOLD,
				color = POLICY_COLORS[row_index],
				horizontal_alignment = Compose.hright,
			),
		)
		push!(
			elements,
			text_context(
				0.26,
				row_center_y + 0.017,
				ROW_LABELS[row_index][2];
				fontsize_points = 12.0,
				color = "#4b5563",
				horizontal_alignment = Compose.hright,
			),
		)

		for metric_index::Int in eachindex(METRIC_LABELS)
			cell_x::Float64 =
				absolute_x_minimum + Float64(metric_index - 1) * metric_width
			value::Float64 = RETENTION_MATRIX[row_index, metric_index]
			background::Colors.RGB{Float64} = retention_color(value)
			push!(
				elements,
				Compose.compose(
					Compose.context(),
					Compose.rectangle(
						cell_x,
						row_y + 0.004,
						metric_width,
						row_height - 0.008,
					),
					Compose.fill(background),
					Compose.stroke("#ffffff"),
					Compose.linewidth(0.45Compose.mm),
				),
			)
			push!(
				elements,
				text_context(
					cell_x + 0.5 * metric_width,
					row_center_y,
					percentage_label(value);
					fontsize_points = 13.2,
					font_name = FONT_BOLD,
					color = text_color(background),
				),
			)
		end

		push!(
			elements,
			box_context(
				balanced_x,
				row_y + 0.004,
				balanced_width,
				row_height - 0.008;
				fill_color = "#eef2ff",
				stroke_color = "#6366f1",
				linewidth_millimeters = 0.38,
			),
		)
		push!(
			elements,
			text_context(
				balanced_x + 0.5 * balanced_width,
				row_center_y,
				percentage_label(BALANCED_SCORES[row_index]);
				fontsize_points = 13.2,
				font_name = FONT_BOLD,
				color = "#3730a3",
			),
		)

		cost_x::Float64 = log_cost_position(
			POSSIBLE_CELL_COSTS[row_index],
			cost_x_minimum,
			cost_x_maximum,
		)
		push!(
			elements,
			path_context(
				[(cost_x_minimum, row_center_y), (cost_x, row_center_y)];
				color = POLICY_COLORS[row_index],
				arrow_at_end = false,
				linewidth_millimeters = 0.75,
			),
		)
		push!(
			elements,
			circle_context(
				cost_x,
				row_center_y,
				0.008;
				fill_color = POLICY_COLORS[row_index],
			),
		)
		if cost_x > cost_x_minimum + 0.72 * (cost_x_maximum - cost_x_minimum)
			push!(
				elements,
				text_context(
					cost_x - 0.012,
					row_center_y - 0.018,
					abbreviated_cost(POSSIBLE_CELL_COSTS[row_index]);
					fontsize_points = 11.2,
					color = "#111827",
					horizontal_alignment = Compose.hright,
				),
			)
		else
			push!(
				elements,
				text_context(
					cost_x + 0.012,
					row_center_y - 0.018,
					abbreviated_cost(POSSIBLE_CELL_COSTS[row_index]);
					fontsize_points = 11.2,
					color = "#111827",
					horizontal_alignment = Compose.hleft,
				),
			)
		end
	end

	legend_x_minimum::Float64 = 0.30
	legend_x_maximum::Float64 = 0.58
	legend_y::Float64 = 0.885
	legend_steps::Int = 50
	for legend_index::Int in 1:legend_steps
		legend_value::Float64 =
			Float64(legend_index - 1) / Float64(legend_steps - 1)
		legend_x::Float64 =
			legend_x_minimum +
			Float64(legend_index - 1) *
			(legend_x_maximum - legend_x_minimum) /
			Float64(legend_steps)
		push!(
			elements,
			Compose.compose(
				Compose.context(),
				Compose.rectangle(
					legend_x,
					legend_y,
					(legend_x_maximum - legend_x_minimum) /
					Float64(legend_steps),
					0.025,
				),
				Compose.fill(retention_color(legend_value)),
				Compose.stroke(retention_color(legend_value)),
			),
		)
	end
	push!(
		elements,
		text_context(
			legend_x_minimum - 0.012,
			legend_y + 0.0125,
			"0%";
			fontsize_points = 11.5,
			horizontal_alignment = Compose.hright,
		),
	)
	push!(
		elements,
		text_context(
			legend_x_maximum + 0.012,
			legend_y + 0.0125,
			"100%";
			fontsize_points = 11.5,
			horizontal_alignment = Compose.hleft,
		),
	)
	push!(
		elements,
		text_context(
			0.5 * (legend_x_minimum + legend_x_maximum),
			legend_y + 0.045,
			"Absolute retained information (fixed intrinsic scale)";
			fontsize_points = 12.5,
			color = "#374151",
		),
	)
	push!(
		elements,
		text_context(
			0.5,
			0.955,
			"* Balanced score is relative to the best of 176 candidates; every heatmap cell is an absolute retained fraction.";
			fontsize_points = 12.5,
			color = "#4b5563",
		),
	)
	return Compose.compose(Compose.context(), reverse(elements)...)
end

function save_figure(
	figure::Compose.Context,
	basename::String,
	width_inches::Float64,
	height_inches::Float64,
)::Nothing
	mkpath(OUTPUT_DIRECTORY)
	svg_path::String = joinpath(OUTPUT_DIRECTORY, "$(basename).svg")
	png_path::String = joinpath(OUTPUT_DIRECTORY, "$(basename).png")
	Compose.draw(
		Compose.SVG(
			svg_path,
			width_inches * Compose.inch,
			height_inches * Compose.inch,
		),
		figure,
	)
	Compose.draw(
		Compose.PNG(
			png_path,
			width_inches * Compose.inch,
			height_inches * Compose.inch;
			dpi = 300,
		),
		figure,
	)
	println("Saved $(svg_path)")
	println("Saved $(png_path)")
	return nothing
end

function main()::Nothing
	framework::Compose.Context = framework_figure()
	policy_summary::Compose.Context = policy_summary_figure()
	save_figure(framework, FRAMEWORK_BASENAME, 10.0, 5.8)
	save_figure(policy_summary, POLICY_BASENAME, 10.0, 6.3)
	return nothing
end

main()
