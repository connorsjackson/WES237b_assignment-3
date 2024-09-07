#define tile_size 16

__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __local float A_kernel[tile_size][tile_size];
  __local float B_kernel[tile_size][tile_size];
  int row_kernel = local(0);
  int col_kernel = local(1);
  int row_main = tile_size * get_group_id(0) + row_kernel(0);
  int col_main = tile_size * group(0) + col_kernel(0);

  int num_tiles = (numAColumns + tile_size) / tile_size;
  float accum = 0;

  for (int tile = 0; tile < num_tiles;){
    tile_row = tile_size * tile + row_kernel;
    tile_col = tile_size * tile + col_kernel;
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