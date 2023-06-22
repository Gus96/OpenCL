#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <omp.h>
#include <iostream>
#include <string>

const int THREADS = 6;
#define BLOCK_SIZE 16

//size_t K = 1280;
//size_t M = 1280;
//size_t N = 1280;
size_t K = 2048;
size_t M = 1920;
size_t N = 1600;


//size_t K = 1584;
//size_t M = 1280;
//size_t N = 1600;
//size_t K = 1280;
//size_t M = 1280;
//size_t N = 1280;



const char* matrixMulGPU = "__kernel void multi (const int n, __global float* a, __global float* b, __global float* c) {  \n"
"int i = get_global_id(1);                                     \n"
"int j = get_global_id(0);                                     \n"
"float sum = 0.0;                                              \n"
"int line = i * n;                                             \n"
"for (int k = 0; k < n; k++) {                                 \n"
"    sum += a[line + k] * b[k * n + j];						   \n"
"}                                                             \n"
"c[line + j] = sum;                                            \n"
"}                                                             \n";


const char* matrixMulGPU_Opti =
"__kernel void multi_opt(const int m, const int n, const int k, __global float* a, __global float* b, __global float* c) {   \n"
"	const int column = get_local_id(0);                            \n"
"	const int row = get_local_id(1);                               \n"
"  const int globalCol = get_global_id(0);						   \n"
"  const int globalRow = get_global_id(1);						   \n"
"	__local float A_m_sub[BLOCK_SIZE][BLOCK_SIZE];                 \n"
"	__local float B_m_sub[BLOCK_SIZE][BLOCK_SIZE];                 \n"
"	float sum = 0.0f;                                              \n"
"	for (int t = 0; t < n / BLOCK_SIZE; t++) {                     \n"
"		const int tiledCol = BLOCK_SIZE * t + column;              \n"
"		const int tiledRow = BLOCK_SIZE * t + row;                 \n"
"		A_m_sub[row][column] = a[globalRow * n + tiledCol];        \n"
"		B_m_sub[row][column] = b[tiledRow * k + globalCol];        \n"
"		barrier(CLK_LOCAL_MEM_FENCE);                              \n"
"		for (int i = 0; i < BLOCK_SIZE; i++) {                     \n"
"			sum += A_m_sub[row][i] * B_m_sub[i][column];           \n"
"		}                                                          \n"
"		barrier(CLK_LOCAL_MEM_FENCE);                              \n"
"	}                                                              \n"
"	c[globalRow * k + globalCol] = sum;                            \n"
"}                                                                 \n";




const char* matrixMulGPUImg =
"kernel void multi_img(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {  \n"
"  const int column = get_local_id(0);                             \n"
"  const int row = get_local_id(1);                                \n"
"  const int globalCol = get_global_id(0);						   \n"
"  const int globalRow = get_global_id(1);						   \n"
"  int n = get_global_size(0);                                     \n"
"  local float A_m_sub[BLOCK_SIZE][BLOCK_SIZE];                    \n"
"  local float B_m_sub[BLOCK_SIZE][BLOCK_SIZE];                    \n"
"  float sum = 0.0f;                                               \n"
"  for (int t = 0; t < n / BLOCK_SIZE; t++) {                      \n"
"    const int tiledCol = BLOCK_SIZE * t + column;                 \n"
"    const int tiledRow = BLOCK_SIZE * t + row;                    \n"
"    const int2 idA = {tiledCol, globalRow};                       \n"
"    const int2 idB = {globalCol, tiledRow};                       \n"
"    A_m_sub[row][column] = read_imagef(a, idA).x;                 \n"
"    B_m_sub[row][column] = read_imagef(b, idB).x;                 \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"    for (int i = 0; i < BLOCK_SIZE; i++) {                        \n"
"      sum += A_m_sub[row][i] * B_m_sub[i][column];                \n"
"     }                                                            \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"  }                                                               \n"
"  const int2 idC = {globalCol, globalRow};                        \n"
"  write_imagef(c, idC, sum);                                      \n"
"}                                                                 \n";




void matrixMul(float* a, float* b, float* c, int k, int m, int n)
{
	int line;
	int column;
	for (int i = 0; i < m; i++) {//идем по строкам
		line = i * n;
		for (int j = 0; j < k; j++) {//идем по столбцам
			column = j * n;
			c[i * k + j] = 0;
			for (int l = 0; l < n; l++) {
				c[i * k + j] += a[line + l] * b[column + l];
			}
		}
	}
}


void matrixMulOMP(float* a, float* b, float* c, int k, int m, int n)
{
	int i, j, l;
	int line;
	int column;
#pragma omp parallel for private(i, j, l, line, column)
	for (i = 0; i < m; i++) {
		line = i * n;
		for (j = 0; j < k; j++) {
			column = j * n;
			c[i * k + j] = 0;
			for (l = 0; l < n; l++) {
				c[i * k + j] += a[line + l] * b[column + l];
			}
		}
	}
}
void printMatrix(float* matrix, int m, int n)
{
	if (n < 10) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				std::cout << matrix[i * n + j] << "\t";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}
void transpose(float* a, float* a_t, const int& n, const int& k) {
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a_t[j * k + i] = a[i * n + j];
		}
	}
}
void OpenCL_task1(float* data_a, float* data_b, float* result, size_t k, size_t m, size_t n, cl_device_id device, cl_platform_id platform, const char* matrixMul, double& time) {
	cl_int error;
	double start, end;

	cl_context_properties properties[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	};


	cl_context context = clCreateContextFromType((NULL == platform) ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateContextFromType" << std::endl;

	size_t size_c = 0;

	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size_c);
	if (error != CL_SUCCESS) std::cout << "Error clGetContextInfo1" << std::endl;

	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size_c, device, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clGetContextInfo2" << std::endl;

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateCommandQueue" << std::endl;



	size_t srclen[] = { strlen(matrixMul) };

	cl_program program = clCreateProgramWithSource(context, 1, &matrixMul, srclen, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateProgramWithSource" << std::endl;

	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, "multi", &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateKernel  " << error << std::endl;

	cl_mem a = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * m * n,
		NULL,
		NULL);

	cl_mem b = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * n * k,
		NULL,
		NULL);

	cl_mem c = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * m * k,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer(
		queue,
		a,
		CL_TRUE,
		0,
		sizeof(float) * m * n,
		data_a,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_a failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer(
		queue,
		b,
		CL_TRUE,
		0,
		sizeof(float) * n * k,
		data_b,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_b failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		0,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		1,
		sizeof(cl_mem),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		2,
		sizeof(cl_mem),
		&b);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		3,
		sizeof(cl_mem),
		&c);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for c failed: " << error << std::endl;
	}

	size_t group = 16;
	clGetKernelWorkGroupInfo(
		kernel,
		device,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t),
		&group,
		NULL);

	cl_event evt;

	size_t* global = new size_t[2];
	global[0] = (size_t)k;
	global[1] = (size_t)n;

	size_t* local = new size_t[2];
	local[0] = (size_t)BLOCK_SIZE;
	local[1] = (size_t)BLOCK_SIZE;

	int dim = 2;

	start = omp_get_wtime();
	error = clEnqueueNDRangeKernel(
		queue,
		kernel,
		dim,
		NULL,
		global,
		local,
		0,
		NULL,
		&evt);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clFinish(queue);
	end = omp_get_wtime();
	time = end - start;

	clEnqueueReadBuffer(
		queue,
		c,
		CL_TRUE,
		0,
		sizeof(float) * n * n,
		result,
		0,
		NULL,
		NULL);

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

}

void OpenCL_task2(float* data_a, float* data_b, float* result, size_t k, size_t m, size_t n, cl_device_id device, cl_platform_id platform, const char* matrixMul, double& time) {
	cl_int error;
	double start, end;


	cl_context_properties properties[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
	};


	cl_context context = clCreateContextFromType((NULL == platform) ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateContextFromType" << std::endl;

	size_t size_c = 0;

	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size_c);
	if (error != CL_SUCCESS) std::cout << "Error clGetContextInfo1" << std::endl;

	//
	error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size_c, device, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clGetContextInfo2" << std::endl;
	//

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateCommandQueue" << std::endl;

	std::string buildOpts = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);

	size_t srclen[] = { strlen(matrixMul) };

	cl_program program1 = clCreateProgramWithSource(context, 1, &matrixMul, srclen, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateProgramWithSource" << std::endl;

	error = clBuildProgram(
		program1,
		1,
		&device,
		buildOpts.c_str(),
		NULL,
		NULL);

	cl_kernel kernel1 = clCreateKernel(program1, "multi_opt", &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateKernel  " << error << std::endl;

	cl_mem a = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * M * N,
		NULL,
		NULL);

	cl_mem b = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * N * K,
		NULL,
		NULL);

	cl_mem c = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * M * K,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer(
		queue,
		a,
		CL_TRUE,
		0,
		sizeof(float) * M * N,
		data_a,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_a failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer(
		queue,
		b,
		CL_TRUE,
		0,
		sizeof(float) * N * K,
		data_b,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_b failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel1,
		0,
		sizeof(int),
		&m);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for m failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel1,
		1,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel1,
		2,
		sizeof(int),
		&k);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for k failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel1,
		3,
		sizeof(cl_mem),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel1,
		4,
		sizeof(cl_mem),
		&b);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel1,
		5,
		sizeof(cl_mem),
		&c);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for c failed: " << error << std::endl;
	}


	size_t group = 16;
	clGetKernelWorkGroupInfo(
		kernel1,
		device,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(size_t),
		&group,
		NULL);

	cl_event evt;

	size_t* global = new size_t[2];
	global[0] = (size_t)k;
	global[1] = (size_t)n;

	size_t* local = new size_t[2];
	local[0] = (size_t)BLOCK_SIZE;
	local[1] = (size_t)BLOCK_SIZE;

	int dim = 2;

	start = omp_get_wtime();
	error = clEnqueueNDRangeKernel(
		queue,
		kernel1,
		dim,
		NULL,
		global,
		local,
		0,
		NULL,
		&evt);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clFinish(queue);
	end = omp_get_wtime();
	time = end - start;


	clEnqueueReadBuffer(
		queue,
		c,
		CL_TRUE,
		0,
		sizeof(float) * n * n,
		result,
		0,
		NULL,
		NULL);

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(program1);
	clReleaseKernel(kernel1);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);


}

void OpenCL_task3(float* data_a, float* data_b, float* result, size_t k, size_t m, size_t n, cl_device_id device, cl_platform_id platform, const char* matrixMul, double& time) {
	cl_int error;
	double start, end;

	cl_context_properties properties[3] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	cl_context context = clCreateContextFromType((NULL == platform) ? NULL : properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error clCreateContextFromType" << std::endl;


	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	size_t size = 0;

	clGetContextInfo(
		context,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size);



	cl_command_queue queue = clCreateCommandQueue(
		context,
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	std::string buildOpts = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);

	size_t srclen[] = { strlen(matrixMul) };

	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&matrixMul,
		srclen,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program,
		1,
		&device,
		buildOpts.c_str(),
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build prog failed" << std::endl;
		size_t logSize = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
		char* log = new char[logSize];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
		std::cout << log;
	}

	cl_kernel kernel = clCreateKernel(program,"multi_img",&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	cl_image_format format = { CL_INTENSITY, CL_FLOAT };

	cl_mem a = clCreateImage2D(
		context,
		CL_MEM_READ_ONLY,
		&format,
		M, N,
		0, NULL,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Set create image for a failed: " << error << std::endl;
	}

	cl_mem b = clCreateImage2D(
		context,
		CL_MEM_READ_ONLY,
		&format,
		N, K,
		0,
		NULL,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Set create image for b failed: " << error << std::endl;
	}

	cl_mem c = clCreateImage2D(
		context,
		CL_MEM_WRITE_ONLY,
		&format,
		M, K,
		0,
		NULL,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Set create image for c failed: " << error << std::endl;
	}


	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { N, N, 1 };

	error = clEnqueueWriteImage(
		queue,
		a,
		CL_TRUE,
		origin,
		region,
		0,
		0,
		data_a,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_a failed: " << error << std::endl;
	}

	error = clEnqueueWriteImage(
		queue,
		b,
		CL_TRUE,
		origin,
		region,
		0,
		0,
		data_b,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_b failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		0,
		sizeof(cl_mem),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		1,
		sizeof(cl_mem),
		&b);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << error << std::endl;
	}

	error = clSetKernelArg(
		kernel,
		2,
		sizeof(cl_mem),
		&c);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for c failed: " << error << std::endl;
	}

	cl_event evt;

	size_t* global = new size_t[2];
	global[0] = (size_t)(N);
	global[1] = (size_t)(N);

	size_t* local = new size_t[2];
	local[0] = (size_t)BLOCK_SIZE;
	local[1] = (size_t)BLOCK_SIZE;

	int dim = 2;

	const size_t offsets[] = { 0, 0 };

	start = omp_get_wtime();
	error = clEnqueueNDRangeKernel(
		queue,
		kernel,
		dim,
		offsets,
		global,
		local,
		0,
		NULL,
		&evt);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clFinish(queue);
	end = omp_get_wtime();
	time = end - start;

	clEnqueueReadImage(
		queue,
		c,
		CL_TRUE,
		origin,
		region,
		0,
		0,
		result,
		0,
		NULL,
		NULL);



	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}





int main() {

	double start_time;
	double end_time;

	double linear_time;
	double omp_time;
	double gpu_time;
	double gpu_opt_time;
	double gpu_img_time;

	float* A = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C = (float*)_aligned_malloc(sizeof(float) * M * K, 64);
	float* B_t = (float*)_aligned_malloc(sizeof(float) * N * K, 64);


	float* A_omp = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_omp = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_omp = (float*)_aligned_malloc(sizeof(float) * M * K, 64);
	float* B_omp_t = (float*)_aligned_malloc(sizeof(float) * N * K, 64);

	float* A_gpu = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_gpu = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_gpu = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_gpu_opt = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_gpu_opt = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_gpu_opt = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_img_gpu = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_img_gpu = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_img_gpu = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float alpha = 1.0f;
	float betta = 4.0f;

	for (int i = 0; i < M * N; i++) {
		A[i] = alpha;
		A_gpu[i] = alpha;
		A_gpu_opt[i] = alpha;
		A_omp[i] = alpha;
		A_img_gpu[i] = alpha;
	}
	for (int i = 0; i < N * K; i++) {
		B[i] = betta;
		B_gpu[i] = betta;
		B_gpu_opt[i] = betta;
		B_omp[i] = betta;
		B_img_gpu[i] = betta;
	}
	for (int i = 0; i < M * K; i++) {
		C[i] = 0.0f;
		C_gpu[i] = 0.0f;
		C_gpu_opt[i] = 0.0f;
		C_omp[i] = 0.0f;
		C_img_gpu[i] = 0.0f;
	}

	//GPU

	cl_int error;
	cl_uint platformCount = 0;
	error = clGetPlatformIDs(0, NULL, &platformCount);
	if (error != CL_SUCCESS) std::cout << "Error clGetPlatformIDs1" << std::endl;
	cl_platform_id* platforms = new cl_platform_id[platformCount];

	error = clGetPlatformIDs(platformCount, platforms, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clGetPlatformIDs2" << std::endl;


	cl_platform_id platform = platforms[0];
	delete[] platforms;

	char platformName[128];
	error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clGetPlatformInfo" << std::endl;

	std::cout << platformName << std::endl;

	cl_uint deviceCount = 0;
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
	if (error != CL_SUCCESS) std::cout << "Error clGetDeviceIDs1" << std::endl;
	cl_device_id* devices = new cl_device_id[deviceCount];
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clGetDeviceIDs2" << std::endl;
	cl_device_id device = devices[0];
	delete[] devices;

	char deviceName[128];
	error = clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, NULL);
	if (error != CL_SUCCESS) std::cout << "Error clGetDeviceInfo1" << std::endl;
	std::cout << deviceName << std::endl;
	std::cout << "________________________________________ " << std::endl;




	//linear
	start_time = omp_get_wtime();
	matrixMul(A, B, C, K, M, N);
	end_time = omp_get_wtime();
	linear_time = end_time - start_time;

	std::cout << "linear time: " << linear_time << std::endl;
	std::cout << "C[10][10]: " << C[10 * M + 10] << std::endl << std::endl;
	std::cout << "________________________________________ " << std::endl;


	//OpenMP
	omp_set_num_threads(THREADS);

	start_time = omp_get_wtime();
	matrixMulOMP(A_omp, B_omp, C_omp, K, M, N);
	end_time = omp_get_wtime();
	omp_time = end_time - start_time;

	std::cout << "omp time: " << omp_time << std::endl;
	std::cout << "C_omp[10][10]: " << C_omp[10 * M + 10] << std::endl << std::endl;
	std::cout << "________________________________________ " << std::endl;









	//PROGRAMS


	OpenCL_task1(A_gpu, B_gpu, C_gpu, K, M, N, device, platform, matrixMulGPU, gpu_time);

	std::cout << "gpu time: " << gpu_time << std::endl;
	std::cout << "C[10][10]: " << C_gpu[10 * M + 10] << std::endl << std::endl;
	std::cout << "________________________________________ " << std::endl;


	OpenCL_task2(A_gpu_opt, B_gpu_opt, C_gpu_opt, K, M, N, device, platform, matrixMulGPU_Opti, gpu_opt_time);

	std::cout << "gpu opt time: " << gpu_opt_time << std::endl;
	std::cout << "C[10][10]: " << C_gpu_opt[10 * M + 10] << std::endl << std::endl;
	std::cout << "________________________________________ " << std::endl;




	OpenCL_task3(A_img_gpu, B_img_gpu, C_img_gpu, K, M, N, device, platform, matrixMulGPUImg, gpu_img_time);

	std::cout << "gpu img time: " << gpu_img_time << std::endl;
	std::cout << "C[10][10]: " << C_img_gpu[10 * M + 10] << std::endl << std::endl;


	return 0;
}