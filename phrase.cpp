#include <CL/cl.h>
#include <iostream>
int main() {
	cl_uint deviceCount = 0;

	const char* source =
		"__kernel void output() {										\n"
		"	printf(														\n"
		"	\"I am from  %d block, %d thread (global index: %d)\\n\",	\n"
		"	(int)get_group_id(0), (int)get_local_id(0), (int)get_global_id(0));		\n"
		"	}															\n";

	//const char* source = "__kernel void output() {printf(\"I am from  %d block, %d thread (global index: %d)\\n\",get_group_id(0), get_local_id(0), get_global_id(0));}";

	cl_uint numPlatforms = 0;//������ ��� ������ ���������� ��������

	clGetPlatformIDs(0, nullptr, &numPlatforms);//����� ���������� ���������� ��������(1)
	std::cout <<"Num platforms: "<< numPlatforms << std::endl;/////////////////////////////////////////////////////////

	cl_platform_id platform = nullptr;
	if (0 < numPlatforms)//���� ���� ���������
	{
		cl_platform_id* platforms = new	cl_platform_id[numPlatforms];//�������� ������ ��� ������ ��������

		clGetPlatformIDs(numPlatforms, platforms, nullptr);//��������� �������� ��������(���-��,����� �� ����������;������ ��� ������ ��������;������ ��� ������ ����� ��������)
		
		for (cl_uint i = 0; i < numPlatforms; ++i)
		{
			char platformName[128];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
				128, platformName, nullptr);
			std::cout << platformName << std::endl;
		}

		//std::cout << platforms[0] << std::endl;
		//std::cout << &platforms[0] << std::endl;
		platform = platforms[0];
		//std::cout << *platform << std::endl;
		delete[] platforms;
	}


	//������� �������� ��������� ��� ������� ���������� ���������
	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	//std::cout << *platform << std::endl;

	cl_int ret;
	cl_context context = clCreateContextFromType(platform == nullptr ? nullptr : properties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &ret);

	//������� �������� � ��������� ���������� ��� ���� ����������� �����������
	size_t size = 0;
	ret = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size);//size - ������� �������� ����

	//�������� ���������� ��� ����������
	cl_device_id device = NULL;//������� NULL
	if (size > 0)
	{
		cl_device_id* devices = (cl_device_id*)alloca(size);
		ret = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, nullptr);//�������� ������ ��������
		device = devices[0];//����� ������� 
		//std::cout << "devices " << size << std::endl;
	}

	char* device_name = new char[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
	std::cout << "Device name: " << device_name << std::endl;

	//������� ������� ������ ��� ��������� ��������� � ���������� ����������
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &ret);

	//������� ����������� ������ �� ��������� ���� (��������� ����)
	size_t srclen[] = { strlen(source) };

	cl_program program = clCreateProgramWithSource(context, 1, &source, srclen, &ret);

	//������� ����������� ���� ��������� ��� ���������� ���������� (���)
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	//������� ������ ����
	cl_kernel kernel = clCreateKernel(program, "output", &ret);


	size_t group;//������������ ������ ������ �����

	ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);
	std::cout << "max group size = " << group << std::endl;

	size_t data_size = 10;
	group = 2;
	std::cout << "selected group size = " << group << std::endl;
	//���������� ���� ��� ���� ���������� ������� ������(������ ���� �� ����������)
	ret = clEnqueueNDRangeKernel(
		queue,
		kernel,
		1,		//����������� ������������ ��������
		nullptr,
		&data_size,/*global work size*/
		&group,/*local work size*/
		0,
		nullptr,
		nullptr);

	ret = clFinish(queue);
	ret = clReleaseProgram(program);
	ret = clReleaseKernel(kernel);
	ret = clReleaseCommandQueue(queue);
	ret = clReleaseContext(context);

	getchar();
	return 0;

}