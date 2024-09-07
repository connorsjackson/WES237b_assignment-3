__kernel void vectorAdd(__global const float *a, __global const float *b,
                        __global float *result, const unsigned int size) {
  //@@ Insert code to implement vector addition here
  int temp = get_global_id(0); //get_global_id() returns 32 bit signed int
  result[temp] = a[temp] + b[temp];
}