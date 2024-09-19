__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

    float sum = 0.0f;
    int row= get_global_id(0); 
    int col = get_global_id(1);


    printf("kernel :  Global ID: (%d, %d)\n",get_global_id(0), get_global_id(1));

        // Perform computation for this tile
        for (int k = 0; k <numAColumns ; k++) {
            sum += A[numAColumns*row + k] * B[numBColumns*k+col];
        }   

    // Store the final result
    
    C[row * numCColumns + col] = sum;
    
    
}