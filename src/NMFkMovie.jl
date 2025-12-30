import FFMPEG
import ImageMagick
import Colors

function _run_ffmpeg(args::AbstractVector{<:AbstractString}; quiet::Bool)
	FFMPEG.ffmpeg() do exe
		cmd = `$exe $(args...)`
		if quiet
			run(pipeline(cmd, stdout=devnull, stderr=devnull))
		else
			run(cmd)
		end
	end
end

function _movie_command_args(format::AbstractString, input_args::Vector{String}, output_base::AbstractString, vspeed::Number, stop_duration::Number)
	fmt = lowercase(strip(format))
	supported = Set(["avi", "webm", "gif", "mp4"])
	if fmt ∉ supported
		@warn "Unknown movie format $format; mp4 will be used!"
		fmt = "mp4"
	end
	setpts_expr = "setpts=$(vspeed)*PTS"
	args = copy(input_args)
	if fmt == "avi"
		append!(args, ["-vcodec", "png", "-filter:v", setpts_expr, "-y", "$output_base.avi"])
	elseif fmt == "webm"
		append!(args, ["-vcodec", "libvpx", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-filter:v", setpts_expr, "-y", "$output_base.webm"])
	elseif fmt == "gif"
		append!(args, ["-f", "gif", "-filter:v", setpts_expr, "-y", "$output_base.gif"])
	else
		filter_expr = "scale=trunc(iw/2)*2:trunc(ih/2)*2,$setpts_expr,tpad=stop_mode=clone:stop_duration=$stop_duration"
		append!(args, ["-filter:v", filter_expr, "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "30", "-r", "30", "-y", "$output_base.mp4"])
		fmt = "mp4"
	end
	return fmt, args
end

"vspeed = 10 - ten times slower; vspped=0.1 - ten times faster"
function makemovie(prefix::AbstractString; movieformat::AbstractString="mp4", movieopacity::Bool=false, moviedir::AbstractString=".", imgformat::AbstractString="png", cleanup::Bool=false, quiet::Bool=true, vspeed::Number=1.0, frame_padding_digits::Integer=0, frame_order::Symbol=:alphanumeric)
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
	if movieopacity
		if isempty(frame_files)
			frame_files = _find_frame_files(moviedir, prefix, imgformat)
		end
		if isempty(frame_files)
			@warn "No frames matching $(prefix) found in $(moviedir); skipping opacity conversion."
			return
		end
		for f in frame_files
			e = splitext(f)
			_convert_with_background(f, string(e[1], ".jpg"))
		end
		cleanup && _remove_files(frame_files)
		imgformat = "jpg"
		frame_files = _find_frame_files(moviedir, prefix, imgformat)
	end
	stop_duration = vspeed / 25
	concat_file = nothing
	input_args = String[]
	if frame_padding_digits > 0
		frame_pattern = "$p%0$(frame_padding_digits)d.$imgformat"
		input_args = ["-i", frame_pattern]
	else
		if isempty(frame_files)
			frame_files = _find_frame_files(moviedir, prefix, imgformat)
		end
		if isempty(frame_files)
			@warn "No frames matching $(prefix)*.$imgformat found in $(moviedir); aborting movie generation."
			return
		end
		order = frame_order in (:alphanumeric, :timestamp) ? frame_order : begin
			@warn "Unknown frame_order $(frame_order); defaulting to :alphanumeric."
			:alphanumeric
		end
		_sort_frame_files!(frame_files, order)
		concat_file = _write_concat_list(frame_files)
		input_args = ["-f", "concat", "-safe", "0", "-i", concat_file]
	end
	movieformat, ffmpeg_args = _movie_command_args(movieformat, input_args, p, vspeed, stop_duration)
	try
		_run_ffmpeg(ffmpeg_args; quiet=quiet)
	finally
		if !isnothing(concat_file)
			rm(concat_file; force=true)
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
	for (root, _, names) in walkdir(moviedir)
		for name in names
			if startswith(name, prefix) && endswith(name, suffix)
				push!(files, abspath(root, name))
			end
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

function _write_concat_list(files::Vector{String})
	list_path, io = mktemp()
	for file in files
		println(io, "file '", _escape_concat_path(file), "'")
	end
	close(io)
	return list_path
end

function _escape_concat_path(path::AbstractString)
	sanitized = replace(path, "\\" => "/")
	return replace(sanitized, "'" => "'\\''")
end

function _remove_files(files::Vector{String})
	for file in files
		isfile(file) && rm(file; force=true)
	end
	return nothing
end