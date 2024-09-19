__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

    float sum = 0.0f;
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);


    printf("kernel :  Global ID: (%d, %d)\n",get_global_id(0), get_global_id(1));

    __local float Asub[TSIZE][TSIZE];  // Maximum tile size, will use only what's needed
    __local float Bsub[TSIZE][TSIZE];

        // Perform computation for this tile
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load one tile of A and B into local memory
        if (row < M && t * TILE_SIZE + localCol < K)
            Asub[localRow][localCol] = A[row * K + t * TILE_SIZE + localCol];
        else
            Asub[localRow][localCol] = 0.0f;

        if (t * TILE_SIZE + localRow < K && col < N)
            Bsub[localRow][localCol] = B[(t * TILE_SIZE + localRow) * N + col];
        else
            Bsub[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }  

    // Store the final result
    
    C[row * numCColumns + col] = sum;
    
    
}