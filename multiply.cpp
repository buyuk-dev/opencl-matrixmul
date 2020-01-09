// Michał Michalski
//
// Square matrix multiplication SGEMM
//
// g++ -std=c++17 -o multiply -lOpenCL


#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <CL/cl.h>


using namespace std;


const int MATRIX_SIZE = 3;
const int WORK_GROUP_SIZE = 32;
const char KERNEL_PATH[] = "multiply.cl";


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


int main(int argc, char* argv[]) {
    string kernelSource = load_kernel_source(KERNEL_PATH);
    if (kernelSource.size() == 0) {
        cerr << "Failed to load kernel from " << KERNEL_PATH << endl;
        return 1;
    }
    cerr << "Loaded kernel source." << endl;

    // Initialize matrices with some test data.
    vector<float> A(MATRIX_SIZE * MATRIX_SIZE);
    vector<float> B(MATRIX_SIZE * MATRIX_SIZE);
    vector<float> C(MATRIX_SIZE * MATRIX_SIZE);

    for (int i = 0; i < A.size(); i++) { A[i] = 1.0; }
    for (int i = 0; i < B.size(); i++) { B[i] = 2.0; }
    for (int i = 0; i < C.size(); i++) { C[i] = 0.0; }

    // Initialize OpenCL context.
    cerr << "Initialize OpenCL" << endl;
    cl_platform_id platform = 0;
    cl_int ret = clGetPlatformIDs(1, &platform, nullptr);
    cerr << "Number of available platforms: " << platform << endl;

    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    cerr << "Number of available devices: " << device << endl;

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    const int kDeviceNameMaxSize = 1024;
    char deviceName[kDeviceNameMaxSize];
    clGetDeviceInfo(device, CL_DEVICE_NAME, kDeviceNameMaxSize, deviceName, nullptr);
    cerr << "Device name is " << deviceName << endl;

    const char* kernelsList[] = { kernelSource.c_str() };
    const size_t kernelsSizeList[] = {kernelSource.size() };
    cl_program program = clCreateProgramWithSource(context, 1, kernelsList, kernelsSizeList, nullptr);
    clBuildProgram(program, 1, nullptr, "", nullptr, nullptr);

    cl_event event = nullptr;

    // Print compile errors if occured.
    size_t logSize = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

    char* messages = new char[1 + logSize];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, nullptr);
    messages[logSize] = '\0';

    if (logSize > 10) {
        cerr << "Compilation errors: " <<  messages << endl;
        return 1;
    }
    delete messages;

    // Create buffers on device.
    cerr << "Allocate buffers on device." << endl;
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  A.size() * sizeof(float), nullptr, nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  B.size() * sizeof(float), nullptr, nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, C.size() * sizeof(float), nullptr, nullptr);

    // Copy matrices to the device memory.
    cerr << "Copy input data to the device." << endl;
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, A.size() * sizeof(float), A.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, B.size() * sizeof(float), B.data(), 0, nullptr, nullptr);

    // Set kernel arguments.
    cerr << "Configure kernel arguments." << endl;
    cl_kernel kernel = clCreateKernel(program, "MultiplyMatrices", nullptr);
    clSetKernelArg(kernel, 0, sizeof(int), static_cast<const void*>(&MATRIX_SIZE));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), static_cast<void*>(&bufA));
    clSetKernelArg(kernel, 2, sizeof(cl_mem), static_cast<void*>(&bufB));
    clSetKernelArg(kernel, 3, sizeof(cl_mem), static_cast<void*>(&bufC));

    // Add task to the queue.
    cerr << "Execute task." << endl;
    const size_t local[2] = { WORK_GROUP_SIZE, WORK_GROUP_SIZE };
    const size_t global[2] = { MATRIX_SIZE, MATRIX_SIZE };
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, &event);

    // Wait until completed and read results.
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, C.size() * sizeof(float), C.data(), 0, nullptr, nullptr);

    // Clean up OpenCL objects.
    cerr << "Clean up." << endl;
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    cout << "RESULT" << endl;
    for (int i=0; i<MATRIX_SIZE; ++i) {
        for (int j=0; j<MATRIX_SIZE; ++j) {
            cout << C[i * MATRIX_SIZE + j] << " ";
        }
        cout << endl;
    }

    return 0;
}

// =================================================================================================
