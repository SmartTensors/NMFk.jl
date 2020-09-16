"vspeed = 10 - ten times slower; vspped=0.1 - ten times faster"
function makemovie(; movieformat="mp4", movieopacity::Bool=false, moviedir::String=".", prefix::String="", keyword::String="frame", imgformat::String="png", cleanup::Bool=true, quiet::Bool=true, vspeed::Number=1.0, numberofdigits::Integer=6)
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
			c = `convert -background black -flatten -format jpg $(joinpath(s[1], f)) $(joinpath(s[1], e[1])).jpg`
			if quiet
				run(pipeline(c, stdout=devnull, stderr=devnull))
			else
				run(c)
			end
		end
		cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".$imgformat -delete`)
		imgformat = "jpg"
	end
	# c = `ffmpeg -i $p-$(keyword)%06d.png -vcodec png -pix_fmt rgba -f mp4 -filter:v "setpts=$vspeed*PTS" -y $p.mp4`
	if movieformat == "avi"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vcodec png -filter:v "setpts=$vspeed*PTS" -y $p.avi`
	elseif movieformat == "webm"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vcodec libvpx -pix_fmt yuva420p -auto-alt-ref 0 -filter:v "setpts=$vspeed*PTS" -y $p.webm`
	elseif movieformat == "gif"
		c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -f gif -filter:v "setpts=$vspeed*PTS" -y $p.gif`
	elseif movieformat == "mp4"
		s = "ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -filter:v scale=\"trunc(iw/2)*2:trunc(ih/2)*2,setpts=$vspeed*PTS\" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 -y $p.mp4"
		c = `bash -l -c "$s"`
	else
		@warn("Unknown movie format $movieformat; mp4 will be used!")
		movieformat = "mp4"
		# c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 -filter:v "setpts=$vspeed*PTS" -y $p.mp4`
		# c = `ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -vf scale="trunc(iw/2)*2:trunc(ih/2)*2,setpts=$vspeed*PTS" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 -y $p.mp4`
		s = "ffmpeg -i $p-$(keyword)%0$(numberofdigits)d.$imgformat -filter:v scale=\"trunc(iw/2)*2:trunc(ih/2)*2,setpts=$vspeed*PTS\" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 -y $p.mp4"
		c = `bash -l -c "$s"`
	end
	if quiet
		run(pipeline(c, stdout=devnull, stderr=devnull))
	else
		run(c)
	end
	cleanup && run(`find $moviedir -name $prefix-$(keyword)"*".$imgformat -delete`)
	return "$p.$movieformat"
end

function moviehstack(movies...; vspeed::Number=1.0, newname="results"=>"all")
	nm = length(movies)
	moviesall = nothing
	for m = 1:nm
		if occursin(newname[1], movies[m])
			moviesall = replace(movies[m], newname)
			break
		end
	end
	recursivemkdir(moviesall; filename=true)
	if moviesall != nothing
		c = "ffmpeg"
		v = ""
		z = ""
		for m = 1:nm
			c *= " -i $(movies[m])"
			v *= "[$(m-1):v]setpts=$(vspeed)*PTS[v$m];"
			z *= "[v$m]"
		end
		c *= " -filter_complex \"$(v) $(z)hstack=inputs=$(nm)[v]\" -map \"[v]\" $(moviesall) -y"
		run(`bash -l -c "$c"`)
		return moviesall
	else
		@warn("Movie filenames cannot be renamed $(newname)!")
	end
end

function movievstack(movies...; vspeed::Number=1.0, newname="results"=>"all")
	nm = length(movies)
	moviesall = nothing
	for m = 1:nm
		if occursin(newname[1], movies[m])
			moviesall = replace(movies[m], newname)
			break
		end
	end
	recursivemkdir(moviesall; filename=true)
	if moviesall != nothing
		c = "ffmpeg"
		v = ""
		z = ""
		for m = 1:nm
			c *= " -i $(movies[m])"
			v *= "[$(m-1):v]setpts=$(vspeed)*PTS[v$m];"
			z *= "[v$m]"
		end
		c *= " -filter_complex \"$(v) $(z)vstack=inputs=$(nm)[v]\" -map \"[v]\" $(moviesall) -y"
		run(`bash -l -c "$c"`)
		return moviesall
	else
		@warn("Movie filenames cannot be renamed $(newname)!")
	end
end