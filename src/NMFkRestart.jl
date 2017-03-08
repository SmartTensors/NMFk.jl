execute_singlerun_r3 = ReusableFunctions.maker3function(NMFk.execute_singlerun_compute, joinpath(pwd(), "restart"))

"Restart on"
function restarton()
    global restart = true;
end

"Restart off"
function restartoff()
    global restart = false;
end