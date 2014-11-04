#define __CL_ENABLE_EXCEPTIONS

#include <OpenCL/opencl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

typedef float element_type;
typedef std::vector<element_type> matrix;

int main()
{
	std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0]);

      // load opencl source
      std::ifstream cl_file("convolution_2D.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);

      // create a message to send to kernel
      
      
		std::ifstream inputFile("input.txt");
		size_t input_width = 0;
		size_t mask_width = 0;
		inputFile >> input_width >> mask_width;
      size_t input_size = input_width * input_width;
      size_t mask_size = mask_width * mask_width;
      matrix input(input_size);
      matrix mask(mask_size);
      size_t result_size = input_size;
      matrix result(result_size);
		for (size_t i = 0; i < input_size; i++) {
         inputFile >> input[i];
      }
      for (size_t i = 0; i < mask_size; i++) {
         inputFile >> mask[i];
      }

       

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(element_type) * input_size);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(element_type) * mask_size);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(element_type) * result_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(element_type) * input_size, &input[0]);
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(element_type) * mask_size, &mask[0]);

      // load named kernel from opencl source
		cl::Kernel kernel(program, "convolution_2D");
		cl::KernelFunctor convolution_2D(kernel, queue, cl::NullRange, cl::NDRange(input_width, input_width), cl::NullRange);
		cl::Event event = convolution_2D(dev_a, dev_b, dev_c, (int)input_width, (int)mask_width);
      event.wait();

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(element_type) * result_size, &result[0]);
		
      //print result
      std::ofstream outputFile("output.txt");
      for (size_t i = 0; i < result_size; i++) {
         outputFile << result[i] << " ";
         if ((i + 1) % input_width == 0)
            outputFile << std::endl;
      }
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}