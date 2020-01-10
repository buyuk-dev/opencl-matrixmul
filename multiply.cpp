// Michał Michalski
//
// Square matrix multiplication SGEMM
//
// Compilation on linux:
//   g++ -std=c++17 -o multiply multiply.cpp -lOpenCL
//
// Compilation on macos:
//   g++ -std=c++17 -o multiply multiply.cpp -framework OpenCL

#define CL_SILENCE_DEPRECATION

#include "utils.hpp"

#include <iostream>
#include <vector>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif // __APPLE__

using namespace std;


const int MATRIX_SIZE = 1024;
const int WORK_GROUP_SIZE = 8;
const char KERNEL_PATH[] = "multiply.cl";


void multiplyOnCPU(const int SIZE, const vector<float>& A, const vector<float>& B, vector<float>& C) {
    for (int row = 0; row < SIZE; ++row) {
        for (int col = 0; col < SIZE; ++col) {
            float sum = 0.0f;
            for (int idx = 0; idx < SIZE; ++idx) {
                sum += A[idx * SIZE + row] * B[col * SIZE + row];
            }
            C[row * SIZE + col] = sum;
        }
    }
}


int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    string kernelSource = loadKernelSource(KERNEL_PATH);
    if (kernelSource.size() == 0) {
        cerr << "Failed to load kernel from " << KERNEL_PATH << endl;
        return 1;
    }

    // Initialize matrices with some test data.
    const int kSquareSize = MATRIX_SIZE * MATRIX_SIZE;
    vector<float> C(kSquareSize);

    vector<float> A(kSquareSize);
    for (int i = 0; i < A.size(); i++) { A[i] = 3.0; }

    vector<float> B(kSquareSize);
    for (int i = 0; i < B.size(); i++) { B[i] = 2.0; }

    // Initialize OpenCL context.
    cl_platform_id platform = 0;
    cl_int ret = clGetPlatformIDs(1, &platform, nullptr);
    cerr << "clGetPlatformIDs() --> " << getErrorString(ret) << endl;

    cl_device_id device = 0;
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    cerr << "clGetDeviceIDs() --> " << getErrorString(ret) << endl;

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    cerr << "clCreateContext() --> " << getErrorString(ret) << endl;

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &ret);
    cerr << "clCreateCommandQueue() --> " << getErrorString(ret) << endl;

    const int kDeviceNameMaxSize = 1024;
    char deviceName[kDeviceNameMaxSize];
    ret = clGetDeviceInfo(device, CL_DEVICE_NAME, kDeviceNameMaxSize, deviceName, nullptr);
    cerr << "clGetDeviceInfo() --> " << getErrorString(ret) << endl;
    cerr << "Selected device is " << deviceName << endl;

    const char* kernelsList[] = { kernelSource.c_str() };
    const size_t kernelsSizeList[] = {kernelSource.size() };
    cl_program program = clCreateProgramWithSource(context, 1, kernelsList, kernelsSizeList, nullptr);
    ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cout << "clBuildProgram() --> " << getErrorString(ret) << endl;

    // Create buffers on device.
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  A.size() * sizeof(float), nullptr, &ret);
    cout << "clCreateBuffer(A) --> " << getErrorString(ret) << endl;

    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  B.size() * sizeof(float), nullptr, &ret);
    cout << "clCreateBuffer(B) --> " << getErrorString(ret) << endl;

    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, C.size() * sizeof(float), nullptr, &ret);
    cout << "clCreateBuffer(C) --> " << getErrorString(ret) << endl;

    // Copy matrices to the device memory.
    cerr << "Copy input data to the device." << endl;
    ret = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, A.size() * sizeof(float), A.data(), 0, nullptr, nullptr);
    cerr << "clEnqueueWriteBuffer(A) --> " << getErrorString(ret) << endl;

    ret = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, B.size() * sizeof(float), B.data(), 0, nullptr, nullptr);
    cerr << "clEnqueueWriteBuffer(B) --> " << getErrorString(ret) << endl; 

    // Set kernel arguments.
    cl_kernel kernel = clCreateKernel(program, "MultiplyMatrices", &ret);
    cerr << "clCreateKernel() --> " << getErrorString(ret) << endl;
    
    ret = clSetKernelArg(kernel, 0, sizeof(int), static_cast<const void*>(&MATRIX_SIZE));
    cerr << "clSetKernelArg(0) --> " << getErrorString(ret) << endl;

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), static_cast<void*>(&bufA));
    cerr << "clSetKernelArg(1) --> " << getErrorString(ret) << endl;

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), static_cast<void*>(&bufB));
    cerr << "clSetKernelArg(2) --> " << getErrorString(ret) << endl;

    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), static_cast<void*>(&bufC));
    cerr << "clSetKernelArg(3) --> " << getErrorString(ret) << endl;

    // Add task to the queue.
    auto t1 = std::chrono::high_resolution_clock::now();

    if (args.find("--gpu") != args.end()) {
        const size_t local[2] = { WORK_GROUP_SIZE, WORK_GROUP_SIZE };
        const size_t global[2] = { MATRIX_SIZE, MATRIX_SIZE };
        ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
        cerr << "clEnqueueNDRangeKernel() --> " << getErrorString(ret) << endl;

        // Read results.
        ret = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, C.size() * sizeof(float), C.data(), 0, nullptr, nullptr);
        cerr << "clEnqueueReadBuffer() --> " << getErrorString(ret) << endl;

        ret = clFlush(queue);
        cerr << "clFlush() --> " << getErrorString(ret) << endl;

        clFinish(queue);
        cerr << "clFinish() --> " << getErrorString(ret) << endl;
    }
    else {
        multiplyOnCPU(MATRIX_SIZE, A, B, C);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    cerr << "multiplication took " << duration << " us." << endl;

    if (args.find("--print") != args.end()) {
        cerr << "----------------------" << endl;
        for (int i=0; i<10; ++i) {
            for (int j=0; j<10; ++j) {
                cout << C[i * MATRIX_SIZE + j] << " ";
            }
            cout << endl;
        }
        cerr << "----------------------" << endl;
    }

    // Clean up OpenCL objects.
    ret = clReleaseMemObject(bufA);
    cerr << "clReleaseMemObject(A) --> " << getErrorString(ret) << endl;

    ret = clReleaseMemObject(bufB);
    cerr << "clReleaseMemObject(B) --> " << getErrorString(ret) << endl;

    ret = clReleaseMemObject(bufC);
    cerr << "clReleaseMemObject(C) --> " << getErrorString(ret) << endl;

    ret = clReleaseCommandQueue(queue);
    cerr << "clReleaseCommandQueue() --> " << getErrorString(ret) << endl;

    ret = clReleaseContext(context);
    cerr << "clReleaseContext() --> " << getErrorString(ret) << endl;

    ret = clReleaseProgram(program);
    cerr << "clReleaseProgram() --> " << getErrorString(ret) << endl;

    ret = clReleaseKernel(kernel);
    cerr << "clReleaseKernel() --> " << getErrorString(ret) << endl;

    return 0;
}

