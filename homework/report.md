Naive MM : 

real    0m5.617s
user    0m3.851s
sys     0m1.541s

This is slower than tiled mm. Some how when I comnpared to cPU based mm , this one is slower. I am not sure what I missed !! maybe I coded the naive mm in gpu badly! need to work more so this one becomes faster than the CPU based mm in Assingment 2. 


Tiled MM : 
Tiled MM was significantly faster than the CPU based tiled mm.

real    0m2.264s
user    0m1.164s
sys     0m1.042s

Tile Size and Local Memory Usage
A fundamental aspect of this implementation is the definition of a tile size (TSIZE) set to 16. This tile size determines the dimensions of sub-blocks into which the input matrices are divided. By using a tile size of 16, the kernel ensures that each workgroup processes a 16x16 block of the matrices. This choice strikes a balance between efficient utilization of computational resources and manageable memory usage.

In main.c, the determination of efficient local work sizes is crucial for maximizing parallelism while adhering to hardware constraints. The get_efficient_local_work_size function calculates the optimal dimensions for workgroups based on the matrix size and the maximum allowed workgroup size (MAX_WORK_GROUP_SIZE set to 1024). 

Kernel Execution and Synchronization
The kernel iterates over each tile, loading corresponding sub-blocks of matrices A and B into local memory. Synchronization barriers (barrier(CLK_LOCAL_MEM_FENCE)) ensure that all work-items within a workgroup have completed loading their respective tiles before proceeding with the multiplication. This synchronization is essential to maintain data consistency and correctness during parallel execution.


