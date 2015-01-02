#define __CL_ENABLE_EXCEPTIONS

#include <OpenCL/opencl.h>
#include "cl.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <cmath>

typedef float element_type;
typedef std::vector<element_type> array;
static const size_t block_size = 256;

struct GpuContext {
	cl::Context& context;
	cl::CommandQueue& queue;
	cl::Program& program;
	GpuContext(cl::Context& c, cl::CommandQueue& q, cl::Program& p) 
		: context(c)
		, queue(q)
		, program(p) 
	{}
};

size_t roundToBlockSize(size_t val) {
	return std::ceil((double)val / block_size);
}

void scan(GpuContext& gpuContext, cl::Buffer& in, cl::Buffer& out, size_t input_size) {
	size_t blocks_num = roundToBlockSize(input_size);
	// allocate auxiliary buffer
	cl::Buffer dev_aux(gpuContext.context, CL_MEM_READ_WRITE,  sizeof(element_type) * blocks_num);
	size_t rounded_size = blocks_num * block_size;
	// load named kernel from opencl source
	cl::Kernel kernel_b(gpuContext.program, "scan_blelloch");
	cl::KernelFunctor scan_b(kernel_b, gpuContext.queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(block_size));
    cl::Event event = scan_b(in, out, dev_aux, cl::__local(sizeof(element_type) * block_size), input_size);
    event.wait();
	
    if (blocks_num == 1) return;
   
   	// scan aux buffer
	scan(gpuContext, dev_aux, dev_aux, blocks_num);
	
	// add aux buffer to out
	cl::Kernel kernel_add(gpuContext.program, "add_aux");
	cl::KernelFunctor add_aux(kernel_add, gpuContext.queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(block_size));
    cl::Event addEvent = add_aux(dev_aux, out, input_size);
    addEvent.wait(); 
}


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
      std::ifstream cl_file("scan.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);
      // compile opencl source
      program.build(devices);
      
      // read input file
      std::ifstream inputFile("input.txt");
	  size_t input_size;
      inputFile >> input_size;
	  
      array input(input_size);
      size_t result_size = input_size;
      array result(result_size);
      for (size_t i = 0; i < input_size; i++) {
      	inputFile >> input[i];
      } 
	  
	    

      // allocate device buffer to hold message
      cl::Buffer dev_in(context, CL_MEM_READ_ONLY,  sizeof(element_type) * input_size);
      cl::Buffer dev_out(context, CL_MEM_READ_WRITE, sizeof(element_type) * result_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_in, CL_TRUE, 0, sizeof(element_type) * input_size, &input[0]);	
	  GpuContext gpuContext(context, queue, program);
	  scan(gpuContext, dev_in, dev_out, input_size);
      queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, sizeof(element_type) * result_size, &result[0]);
		
      //print result
      std::ofstream outputFile("output.txt");
      for (size_t i = 0; i < result_size; i++) {
		  outputFile << result[i] << " ";
      }
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }
   
   return 0;
}
