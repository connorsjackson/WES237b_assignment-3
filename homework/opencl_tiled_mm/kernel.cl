#define tile_size 16

__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  __local float A_kernel[tile_size][tile_size];
  __local float B_kernel[tile_size][tile_size];
  int row_kernel = get_local_id(0);
  int col_kernel = get_local_id(1);
  int row_main = tile_size * get_group_id(0) + row_kernel;
  int col_main = tile_size * get_group_id(1) + col_kernel;

  int num_tiles = (numAColumns + tile_size) / tile_size; //overestimate; may allocated threads we won't use
  float accum = 0;

  //Now we loop thru the tiles on the outside.
  for (int tile = 0; tile < num_tiles;tile++){
    //loop within the kernel
    for (int t = 0; t < tile_size; t++){

      //convert local to global locations
      int tile_row = tile_size * tile + t;
      int tile_col = tile_size * tile + t;
      
      //populate kernel matrices ("kernel matrices" are the small tiles)
      if( (row_main < numARows) && (tile_col < numAColumns) ){ //if (row <#ofrows && col < #ofcols)
        A_kernel[row_kernel][t] = A[row_main*numAColumns + tile_col];
      }
      else{
        A_kernel[row_kernel][t] = 0;
      }
      if( (tile_row < numBRows) && (col_main < numBColumns)){
        B_kernel[t][col_kernel] = B[tile_row*numBColumns + col_main];
      }
      else{
        B_kernel[t][col_kernel] = 0;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); //"block" until all threads reach this point

    //populate the accumulator
    for (int k = 0; k < tile_size; k++){
      accum += A_kernel[row_kernel][k] * B_kernel[k][col_kernel];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ( (row_main < numCRows) && (col_main < numCColumns)){
      C[row_main*numCColumns + col_main] = accum;
    }
  }
}

/* Naive MM is below
for (int i = 0; i< A_row ; i++){
  for (int j=0; j < B_col; j++){
    for (int k=0; k<A_col; k++){
      // C[x,y] = A[x,k] + B[k,y]
      C[i*A_row + j] += A[i*A_col + k] + B[k*B_col + j]
    }
  }
}
*/