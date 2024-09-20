#define TSIZE 16

__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns
    //const unsigned int TSIZE
    ) {
    
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    const int localRow = get_local_id(1);
    const int localCol = get_local_id(0);

    __local float Asub[TSIZE][TSIZE];
    __local float Bsub[TSIZE][TSIZE];

    float sum = 0.0f;

    // Loop over all tiles
    const int numTiles = (numAColumns + TSIZE - 1) / TSIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into local memory
        const int tiledRow = TSIZE * t + localRow;
        const int tiledCol = TSIZE * t + localCol;
        
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
        
        //Synchronize all locals 
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform computation for this tile
        for (int k = 0; k < TSIZE; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
            //sum = native_fma(Asub[localRow][k], Bsub[k][localCol], sum);
        }
        //Synchronize all locals 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result
    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = sum;
    }
}