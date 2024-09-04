__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  
  //@@ Insert code to implement matrix multiplication here
  
  int i = get_global_id(0);
  int j = get_global_id(1);
  for (int k=0; k < numAColumns; k++){
    C[i*numCColumns +  j] += (A[i*numAColumns+k] * B[k*numBColumns + j]);
  }
}

/* Naive MM is below
for (int i = 0; i< A_row ; i++){
  for (int j=0; j < B_col; j++){
    for (int k=0; k<A_col; k++){
      // C[x,y] = A[x,k] + B[k,y]
      C[i*A_col + k] += A[i*A_col + k] + B[k*B_col + j]
    }
  }
}
*/