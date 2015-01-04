#define __CL_ENABLE_EXCEPTIONS

#include <OpenCL/opencl.h>
#include "cl.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <string>
#include <sstream>

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

template<typename T>
void scan(GpuContext& gpuContext, cl::Buffer& in, cl::Buffer& out, size_t input_size, const std::string& type_name) {
    size_t blocks_num = roundToBlockSize(input_size);
    // allocate auxiliary buffer
    cl::Buffer dev_aux(gpuContext.context, CL_MEM_READ_WRITE,  sizeof(T) * blocks_num);
    size_t rounded_size = blocks_num * block_size;
    // load named kernel from opencl source
    std::string scan_name = std::string("scan_blelloch_") + type_name;
    cl::Kernel kernel_b(gpuContext.program, scan_name.c_str());
    cl::KernelFunctor scan_b(kernel_b, gpuContext.queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(block_size));
    cl::Event event = scan_b(in, out, dev_aux, cl::__local(sizeof(T) * block_size), input_size);
    event.wait();
    
    if (blocks_num == 1) return;
   
    cl::Buffer dev_scan_aux(gpuContext.context, CL_MEM_READ_WRITE,  sizeof(T) * blocks_num);
    // scan aux buffer
    scan<T>(gpuContext, dev_aux, dev_scan_aux, blocks_num, type_name);
    
    // add aux buffer to out
    std::string add_name = std::string("add_aux_") + type_name;
    cl::Kernel kernel_add(gpuContext.program, add_name.c_str());
    cl::KernelFunctor add_aux(kernel_add, gpuContext.queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(block_size));
    cl::Event addEvent = add_aux(dev_scan_aux, out, input_size);
    addEvent.wait(); 
}

template<typename T>
void exec(std::vector<T>& input, const std::string& type_name) {
    typedef std::vector<T> array;
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
      

	  
        size_t input_size = input.size();
        size_t result_size = input_size;
        array result(result_size);
	  
        // allocate device buffer to hold message
        cl::Buffer dev_in(context, CL_MEM_READ_ONLY,  sizeof(T) * input_size);
        cl::Buffer dev_out(context, CL_MEM_READ_WRITE, sizeof(T) * result_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_in, CL_TRUE, 0, sizeof(T) * input_size, &input[0]);	
        GpuContext gpuContext(context, queue, program);
        scan<T>(gpuContext, dev_in, dev_out, input_size, type_name);
        queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, sizeof(T) * result_size, &result[0]);
		
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
    
    
}


int main()
{
    // read input file
    std::ifstream inputFile("input.txt");
    size_t input_size;
    inputFile >> input_size;
    inputFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string data;
    std::getline(inputFile, data);
    bool is_float = data.find('.') != std::string::npos;
    std::stringstream stream(data);
    if (is_float) {
        std::vector<float> input(input_size);
        for (size_t i = 0; i < input_size; i++) {
            stream >> input[i];
        } 
        exec<float>(input, "float");
    } else {
        std::vector<int> input(input_size);
        for (size_t i = 0; i < input_size; i++) {
            stream >> input[i];
        } 
        exec<int>(input, "int");
    }

    return 0;
}
