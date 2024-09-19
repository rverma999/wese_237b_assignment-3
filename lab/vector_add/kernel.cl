__kernel void vectorAdd(__global const float *a, __global const float *b,
                        __global float *result, const unsigned int size) {
  //@@iInsert code to impleme
  int i=get_global_id(0);
  int j=i;
  float temp_result; //
  if(i>size) return;
  //for (int j=0;j<i;j++) {
    temp_result = a[j]+b[j];
    result[j]= temp_result; 
 // }
  //

  //*result[i] = *a[i]+*b[i];
  //printf("kernel indx=%0d a=%f b=%f result = %f",i,*a[i],*b[i],*result[i]);
  //printf("kernel indx=%0d a=%f b=%f result = %f",i,*a,*b,*result);
}
