#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
using namespace std;


const int LIST_SIZE = 1024;


string load_kernel_source(const std::string& path) {
    ifstream ifs(path);
    if (!ifs) {
        cerr << "file not found: " << path << endl;
        return "";
    }
    stringstream stream;
    stream << ifs.rdbuf();
    return stream.str();
}


vector<int> create_vector(int n) {
    vector<int> v;
    for (int i=0; i<n; ++i) {
        v.push_back(i);
    }
    return v;
}


int main(int argc, char* argv[]) {
    auto A = create_vector(LIST_SIZE);
    auto B = create_vector(LIST_SIZE);
    auto C = vector<int>(LIST_SIZE);
 
    auto kernel_source = load_kernel_source("VectorAdd.cl");
    if (kernel_source.size() == 0) {
        cerr << "Failed to load kernel file.";
        return 1;
    }

    cerr << "OpenCL kernel loaded." << endl;

    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint devices_count, platforms_count;

    cl_int ret = clGetPlatformIDs(1, &platform_id, &platforms_count);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &devices_count);

    cerr << "Number of available platforms: " << platforms_count << endl;
    cerr << "Number of available devices: " << devices_count << endl;

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cerr << "Context and command queue created." << endl;

    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), nullptr, &ret);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), nullptr, &ret);
    cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), nullptr, &ret);
    
    cerr << "Device memory buffers allocated." << endl;

    ret = clEnqueueWriteBuffer(command_queue, a_buffer, CL_TRUE, 0, LIST_SIZE * sizeof(int), A.data(), 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(command_queue, b_buffer, CL_TRUE, 0, LIST_SIZE * sizeof(int), B.data(), 0, nullptr, nullptr);

    cerr << "Input data sent to the device." << endl;

    const char* kernels_list[] = { kernel_source.c_str() };
    const size_t kernels_size_list[] = { kernel_source.size() };
    cl_program program = clCreateProgramWithSource(context, 1, kernels_list, kernels_size_list, &ret);
    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    
    cerr << "Kernel program compiled." << endl;

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), static_cast<void*>(&a_buffer));
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), static_cast<void*>(&b_buffer));
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), static_cast<void*>(&c_buffer));

    cerr << "Kernel arguments assigned." << endl;

    size_t global_item_size = LIST_SIZE; // process the entire list
    size_t local_item_size = 64;         // divide work into groups of 64 items
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr,
        &global_item_size, &local_item_size, 0, nullptr, nullptr
    );

    cerr << "Processing tasks enqueued." << endl;

    ret = clEnqueueReadBuffer(command_queue, c_buffer, CL_TRUE, 0, LIST_SIZE * sizeof(int), C.data(), 0, nullptr, nullptr);

    cerr << "Results received from the device." << endl;

    for (int i=0; i<LIST_SIZE; ++i) {
        cout << A[i] << " + " << B[i] << " = " << C[i] << endl;        
    }

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_buffer);
    ret = clReleaseMemObject(b_buffer);
    ret = clReleaseMemObject(c_buffer);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    cerr << "Cleaned up." << endl;
    return 0;
}
