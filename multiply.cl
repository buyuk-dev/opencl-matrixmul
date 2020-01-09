// Michal Michalski
// -----------------------------------
// Square Matrix Multiplication Kernel
// A[N,N] * B[N,N] = C[N,N]
// -----------------------------------

__kernel void MultiplyMatrices(const int SIZE, const __global float* A, const __global float* B, __global float* C)
{
    // get identifiers of current work item which indicates data to be processed.
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    // calculate dot product of A[row, *] and B[*, col]
    float sum = 0.0f;
    for (int idx = 0; idx < K; idx ++) {
        sum += A[idx * SIZE + row] * B[col * SIZE + idx];
    }

    // save result
    C[col * SIZE + row] = sum;
}

