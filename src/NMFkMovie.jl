import FFMPEG
import ImageMagick
import Colors

function _run_ffmpeg(args::AbstractVector{<:AbstractString}; quiet::Bool)
	FFMPEG.ffmpeg() do exe
		cmd = Cmd(vcat(exe, String.(args)))
		try
			if quiet
				run(pipeline(cmd, stdout=devnull, stderr=devnull))
			else
				run(cmd)
			end
		catch e
			# Capture stderr to get better error messages
			io = IOBuffer()
			try
				run(pipeline(cmd, stderr=io))
			catch
				stderr_output = String(take!(io))
				@error "FFmpeg command failed" command=cmd stderr=stderr_output exception=e
				rethrow(e)
			end
		end
	end
end

function _movie_command_args(
	format::AbstractString,
	input_args::Vector{String},
	output_base::AbstractString,
	vspeed::Number,
	stop_duration::Number;
	codec::Union{Nothing, AbstractString}=nothing,
	bitrate::Union{Nothing, AbstractString}=nothing,
	crf::Union{Nothing, Real}=nothing,
	preset::Union{Nothing, AbstractString}=nothing,
	extra_args::AbstractVector{<:AbstractString}=String[]
)
	fmt = lowercase(strip(format))
	supported = Set(["avi", "webm", "gif", "mp4"])
	if fmt ∉ supported
		@warn "Unknown movie format $format; mp4 will be used!"
		fmt = "mp4"
	end
	setpts_expr = "setpts=$(vspeed)*PTS"
	args = copy(input_args)
	if fmt == "avi"
		append!(args, ["-vcodec", something(codec, "png"), "-filter:v", setpts_expr])
		append!(args, String.(extra_args))
		append!(args, ["-y", "$output_base.avi"])
	elseif fmt == "webm"
		append!(args, ["-vcodec", something(codec, "libvpx"), "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-filter:v", setpts_expr])
		append!(args, String.(extra_args))
		append!(args, ["-y", "$output_base.webm"])
	elseif fmt == "gif"
		append!(args, ["-f", "gif", "-filter:v", setpts_expr])
		append!(args, String.(extra_args))
		append!(args, ["-y", "$output_base.gif"])
	else
		filter_expr = "scale=trunc(iw/2)*2:trunc(ih/2)*2,$setpts_expr,tpad=stop_mode=clone:stop_duration=$stop_duration"
		append!(args, ["-filter:v", filter_expr])
		codec_val = something(codec, "libx264")
		append!(args, ["-c:v", codec_val])
		append!(args, ["-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "30", "-r", "30"])
		if bitrate !== nothing
			append!(args, ["-b:v", String(bitrate)])
		end
		crf_val = something(crf, 20)
		append!(args, ["-crf", string(crf_val)])
		preset_val = something(preset, "slow")
		append!(args, ["-preset", String(preset_val)])
		append!(args, ["-movflags", "+faststart"])
		append!(args, String.(extra_args))
		append!(args, ["-y", "$output_base.mp4"])
		fmt = "mp4"
	end
	return fmt, args
end

"""
	makemovie(prefix; movieformat="mp4", movieopacity=false, moviedir=".", imgformat="png",
			  cleanup=false, quiet=true, vspeed=10.0, frame_padding_digits=0,
			  frame_order=:alphanumeric)

Turn a directory of exported frames into a movie, optionally converting transparent PNGs to
opaque JPGs first. Keyword arguments control the processing pipeline:

- `movieformat` — Output container (`mp4`, `avi`, `gif`, `webm`).
 - `movieformat` — Output container (`mp4`, `avi`, `gif`, `webm`). For `mp4`, high-profile H.264 with CRF=20, preset `slow`, and `+faststart` are used by default.
- `movieopacity` — When true, flatten alpha using ImageMagick before encoding.
- `moviedir`/`imgformat` — Location and file type of the source frames.
- `cleanup` — Remove intermediate frames that match the prefix after movie creation.
- `quiet` — Suppress ffmpeg stdout/stderr.
- `vspeed` — Multiplier for ffmpeg `setpts`; values >1 slow the video, <1 speed it up.
- `frame_padding_digits` — Use ffmpeg's sequential pattern (`%05d`) when frames are zero padded.
- `frame_order` — Sorting strategy for non-padded frames (`:alphanumeric` or `:timestamp`).
- `video_codec`, `video_bitrate`, `video_crf`, `video_preset` — Optional overrides for ffmpeg compression settings (default CRF=20, preset=`slow` for mp4).
- `extra_ffmpeg_args` — Additional raw arguments appended ahead of the output path for advanced tuning.
"""
function makemovie(prefix::AbstractString; files::AbstractVector{<:AbstractString}=String[], movieformat::AbstractString="mp4", movieopacity::Bool=false, moviedir::AbstractString=".", imgformat::AbstractString="png", cleanup::Bool=false, quiet::Bool=true, vspeed::Number=10.0, frame_padding_digits::Integer=0, frame_order::Symbol=:alphanumeric, video_codec::Union{Nothing, AbstractString}=nothing, video_bitrate::Union{Nothing, AbstractString}=nothing, video_crf::Union{Nothing, Real}=nothing, video_preset::Union{Nothing, AbstractString}=nothing, extra_ffmpeg_args::AbstractVector{<:AbstractString}=String[])
	if !isempty(files)
		if moviedir == "."
			moviedir, prefix = splitdir(prefix)
			if moviedir == ""
				moviedir = "."
			end
		end
		p = joinpath(moviedir, prefix)
		frame_files = String[]
		if movieopacity || frame_padding_digits == 0
			frame_files = _find_frame_files(moviedir, prefix, imgformat)
		end
		order = frame_order in (:alphanumeric, :timestamp) ? frame_order : begin
			@warn "Unknown frame_order $(frame_order); defaulting to :alphanumeric."
			:alphanumeric
		end
		_sort_frame_files!(frame_files, order)
	else
		frame_files = files
	end
	if movieopacity
		for f in frame_files
			e = splitext(f)
			_convert_with_background(f, string(e[1], ".jpg"))
		end
		cleanup && _remove_files(frame_files)
		imgformat = "jpg"
		frame_files = _find_frame_files(moviedir, prefix, imgformat)
	end
	stop_duration = vspeed / 25
	temp_frame_dir = nothing
	if frame_padding_digits > 0
		frame_pattern = "$p%0$(frame_padding_digits)d.$imgformat"
	else
		frame_pattern, temp_frame_dir, frame_padding_digits = _materialize_frame_sequence(frame_files, imgformat)
	end
	input_args = ["-i", frame_pattern]
	movieformat, ffmpeg_args = _movie_command_args(
		movieformat,
		input_args,
		p,
		vspeed,
		stop_duration;
		codec=video_codec,
		bitrate=video_bitrate,
		crf=video_crf,
		preset=video_preset,
		extra_args=extra_ffmpeg_args
	)
	try
		_run_ffmpeg(ffmpeg_args; quiet=quiet)
	finally
		if !isnothing(temp_frame_dir)
			rm(temp_frame_dir; force=true, recursive=true)
		end
	end
	cleanup && _cleanup_movie_frames(moviedir, prefix, imgformat)
	println("Movie $p.$movieformat created!")
	return
end

function stackmovie(movies...; dir::Symbol=:h, vspeed::Number=1.0, newname="results"=>"all", quiet::Bool=false)
	nm = length(movies)
	if nm == 0
		@warn("No input movies provided; skipping stack")
		return nothing
	end
	movieall = nothing
	for m = 1:nm
		if occursin(newname[1], movies[m])
			movieall = replace(movies[m], newname)
			break
		end
	end
	if isnothing(movieall)
		@warn("Movie filenames cannot be renamed $(newname)!")
		return nothing
	end
	recursivemkdir(movieall; filename=true)
	stack = dir == :h ? "hstack" : dir == :v ? "vstack" : nothing
	if isnothing(stack)
		@warn("Unknown direction! `dir` can be :h or :v only!")
		return nothing
	end
	args = String[]
	for movie in movies
		append!(args, ["-i", movie])
	end
	setpts_parts = ["[$(m-1):v]setpts=$(vspeed)*PTS[v$m];" for m in 1:nm]
	stack_inputs = join(["[v$m]" for m in 1:nm])
	filter_expr = string(join(setpts_parts), " ", stack_inputs, stack, "=inputs=", nm, "[v]")
	append!(args, ["-filter_complex", filter_expr, "-map", "[v]", "-y", movieall])
	_run_ffmpeg(args; quiet=quiet)
	return movieall
end

function moviehstack(movies...; kw...)
	stackmovie(movies...; kw..., dir=:v)
end

function movievstack(movies...; kw...)
	stackmovie(movies...; kw..., dir=:h)
end

function _cleanup_movie_frames(moviedir::AbstractString, prefix::AbstractString, imgformat::AbstractString)
	_remove_files(_find_frame_files(moviedir, prefix, imgformat))
	return nothing
end

function _convert_with_background(src::AbstractString, dest::AbstractString; background::Colors.RGB{Float64}=Colors.RGB{Float64}(0, 0, 0))
	img = ImageMagick.load(src)
	flat_img = map(img) do px
		_blend_with_background(px, background)
	end
	ImageMagick.save(dest, flat_img)
	return dest
end

function _blend_with_background(px, background::Colors.RGB{Float64})
	a = Colors.alpha(px)
	if a >= 1
		return Colors.RGB{Float64}(px)
	elseif a <= 0
		return background
	else
		fg = Colors.RGB{Float64}(px)
		return Colors.RGB{Float64}((1 - a) * background.r + a * fg.r,
			(1 - a) * background.g + a * fg.g,
			(1 - a) * background.b + a * fg.b)
	end
end

function _find_frame_files(moviedir::AbstractString, prefix::AbstractString, imgformat::AbstractString)
	suffix = string(".", imgformat)
	files = String[]
	for name in readdir(moviedir)
		path = joinpath(moviedir, name)
		if isfile(path) && startswith(name, prefix) && endswith(name, suffix)
			push!(files, abspath(path))
		end
	end
	return files
end

function _sort_frame_files!(files::Vector{String}, order::Symbol)
	if order == :timestamp
		sort!(files, by = f -> (stat(f).mtime, basename(f), f))
	else
		sort!(files, by = f -> (basename(f), f))
	end
	return files
end

function _materialize_frame_sequence(files::Vector{String}, imgformat::AbstractString)
	temp_dir = mktempdir()
	digits = length(string(length(files)))
	pattern = joinpath(temp_dir, "frame%0$(digits)d.$imgformat")
	for (idx, src) in enumerate(files)
		frame_name = string("frame", lpad(string(idx), digits, '0'), ".", imgformat)
		dest = joinpath(temp_dir, frame_name)
		cp(src, dest; force=true)
	end
	return pattern, temp_dir, digits
end

function _remove_files(files::Vector{String})
	for file in files
		isfile(file) && rm(file; force=true)
	end
	return nothing
end