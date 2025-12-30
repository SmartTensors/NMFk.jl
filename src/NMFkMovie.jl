import FFMPEG

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

function _movie_command_args(format::AbstractString, input_pattern::AbstractString, output_base::AbstractString, vspeed::Number, stop_duration::Number)
	fmt = lowercase(strip(format))
	supported = Set(["avi", "webm", "gif", "mp4"])
	if fmt ∉ supported
		@warn "Unknown movie format $format; mp4 will be used!"
		fmt = "mp4"
	end
	setpts_expr = "setpts=$(vspeed)*PTS"
	if fmt == "avi"
		args = ["-i", input_pattern, "-vcodec", "png", "-filter:v", setpts_expr, "-y", "$output_base.avi"]
	elseif fmt == "webm"
		args = ["-i", input_pattern, "-vcodec", "libvpx", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-filter:v", setpts_expr, "-y", "$output_base.webm"]
	elseif fmt == "gif"
		args = ["-i", input_pattern, "-f", "gif", "-filter:v", setpts_expr, "-y", "$output_base.gif"]
	else
		filter_expr = "scale=trunc(iw/2)*2:trunc(ih/2)*2,$setpts_expr,tpad=stop_mode=clone:stop_duration=$stop_duration"
		args = ["-i", input_pattern, "-filter:v", filter_expr, "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "30", "-r", "30", "-y", "$output_base.mp4"]
		fmt = "mp4"
	end
	return fmt, args
end

"vspeed = 10 - ten times slower; vspped=0.1 - ten times faster"
function makemovie(; movieformat::AbstractString="mp4", movieopacity::Bool=false, moviedir::AbstractString=".", prefix::AbstractString="", keyword::AbstractString="frame", imgformat::AbstractString="png", cleanup::Bool=true, quiet::Bool=true, vspeed::Number=1.0, numberofdigits::Integer=6)
	if moviedir == "."
		moviedir, prefix = splitdir(prefix)
		if moviedir == ""
			moviedir = "."
		end
	end
	p = joinpath(moviedir, prefix)
	if movieopacity
		s = splitdir(p)
		files = searchdir(Regex(string("$(s[2])-$(keyword)", ".*\\.", imgformat)), s[1])
		for f in files
			e = splitext(f)
			input_path = joinpath(s[1], f)
			output_path = joinpath(s[1], e[1] * ".jpg")
			args = ["-y", "-i", input_path, "-vf", "format=rgba", "-pix_fmt", "yuvj444p", output_path]
			_run_ffmpeg(args; quiet=quiet)
		end
		cleanup && _cleanup_movie_frames(moviedir, prefix, keyword, imgformat)
		imgformat = "jpg"
	end
	stop_duration = vspeed / 25
	frame_pattern = "$p-$(keyword)%0$(numberofdigits)d.$imgformat"
	movieformat, ffmpeg_args = _movie_command_args(movieformat, frame_pattern, p, vspeed, stop_duration)
	_run_ffmpeg(ffmpeg_args; quiet=quiet)
	cleanup && _cleanup_movie_frames(moviedir, prefix, keyword, imgformat)
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

function _cleanup_movie_frames(moviedir::AbstractString, prefix::AbstractString, keyword::AbstractString, imgformat::AbstractString)
	frame_prefix = string(prefix, "-", keyword)
	frame_suffix = string(".", imgformat)
	for (root, _, files) in walkdir(moviedir)
		for file in files
			if startswith(file, frame_prefix) && endswith(file, frame_suffix)
				frame_path = joinpath(root, file)
				isfile(frame_path) && rm(frame_path; force=true)
			end
		end
	end
	return nothing
end