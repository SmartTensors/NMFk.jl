for i=1:10  
   # spawn the process  
   r = @spawn rand()  

   # print out the results  
   @printf("process: %d %f\n", r.where, fetch(r))  
 end  
