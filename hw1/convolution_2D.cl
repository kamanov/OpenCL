__kernel void convolution_2D(__global float * input, __global float * mask, __global float * output, int input_width, int mask_width)
{
   int idx = get_global_id(0);
   int idy = get_global_id(1);

   if (idx >= input_width || idy >= input_width)
      return;

   float res = 0;
   
   for (int i = 0; i < mask_width; ++i)
   {
      int input_idx = (idx + i - mask_width / 2);
      if (input_idx < 0 || input_idx >= input_width)
         continue;
   	for (int j = 0; j < mask_width; ++j)
   	{
      	int input_idy = (idy + j - mask_width / 2);
      	if (input_idy >= 0 && input_idy < input_width)
         	res += input[input_idx + input_idy * input_width] * mask[i + j * mask_width];
   	}
   }
   output[idx + idy * input_width] = res;
}