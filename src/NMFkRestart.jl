"Restart on"
function restarton()
	global execute_singlerun_r3 = ReusableFunctions.maker3function(execute_singlerun_compute, joinpath(pwd(), "restart"))
    global restart = true
end

"Restart off"
function restartoff()
    global restart = false
end