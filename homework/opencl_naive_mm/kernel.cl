__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    const int localRow = get_local_id(1);
    const int localCol = get_local_id(0);

    __local float Asub[16][16];
    __local float Bsub[16][16];

    float sum = 0.0f;
    printf("Global ID: (%d, %d)\n",get_global_id(0), get_global_id(1));

    // Loop over all tiles
    const int numTiles = (numAColumns + 15) / 16;

    for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into local memory
        const int tiledRow = 16 * t + localRow;
        const int tiledCol = 16 * t + localCol;
        
        if (row < numARows && tiledCol < numAColumns) {
            Asub[localRow][localCol] = A[row * numAColumns + tiledCol];
        } else {
            Asub[localRow][localCol] = 0.0f;
        }
        
        if (tiledRow < numBRows && col < numBColumns) {
            Bsub[localRow][localCol] = B[tiledRow * numBColumns + col];
        } else {
            Bsub[localRow][localCol] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform computation for this tile
        for (int k = 0; k < 16; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result
    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = sum;
    }
}