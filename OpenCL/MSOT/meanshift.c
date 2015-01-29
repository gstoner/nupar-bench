/*
 * -- NUPAR: A Benchmark Suite for Modern GPU Architectures
 *    NUPAR - 2 December 2014
 *    Amir Momeni
 *    Northeastern University
 *    NUCAR Research Laboratory
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the Northeastern U.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <sys/time.h>
#include <CL/cl.h>
#include "my_image.h"
#include "my_meanshift.h"


unsigned char *read_buffer(char *file_name, size_t *size_ptr) {


        FILE *f;
        unsigned char *buf;
        size_t size;

        //open file
        f = fopen(file_name, "rb");
        if(!f)
        {
                printf("File open failed!\n");
                return NULL;
        }

        //botain file size
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        fseek(f, 0, SEEK_SET);

        //alloc and read buffer
        buf = malloc(size + 1);
        fread(buf, 1, size, f);
        buf[size] = '\0';

        //return size of buffer
        if(size_ptr)
                *size_ptr = size;

        //return bufffer
        return buf;
}



int read_object_file(char *address, int *num_objects, short **object_x, short **object_y, short **object_w, short **object_h) {

        FILE *object_file;
        char *line = NULL;
        size_t len = 0;
        ssize_t read_lines;
        int i;

        object_file = fopen(address, "r");

        if (object_file == NULL)
                return -1;

        //read start frame number
        getline(&line, &len, object_file);
        int temp = atoi(line);
        memcpy(num_objects, &temp, sizeof(int));

        (*object_x) = (short *)malloc((*num_objects)*sizeof(short));
        (*object_y) = (short *)malloc((*num_objects)*sizeof(short));
        (*object_w) = (short *)malloc((*num_objects)*sizeof(short));
        (*object_h) = (short *)malloc((*num_objects)*sizeof(short));


        //read x, y, w, h of each object
        for (i = 0; i < (*num_objects); i++) {

                getline(&line, &len, object_file);
                temp = atoi(line);
                memcpy(&(*object_x)[i], &temp, sizeof(short));

                getline(&line, &len, object_file);
                temp = atoi(line);
                memcpy(&(*object_y)[i], &temp, sizeof(short));

                getline(&line, &len, object_file);
                temp = atoi(line);
                memcpy(&(*object_w)[i], &temp, sizeof(short));

                getline(&line, &len, object_file);
                temp = atoi(line);
                memcpy(&(*object_h)[i], &temp, sizeof(short));

        }

        return 0;
}



int read_image_file(char *address, char **benchmark, char **output, char **ext, int *first_frame, int *last_frame) {

        FILE *image_file;
        char *line = NULL;
        size_t len = 0;
        ssize_t read_lines;
        int i;
	int size;

        image_file = fopen(address, "r");

        if (image_file == NULL)
                return -1;
        //read start frame number
        size = getline(&line, &len, image_file);
	printf("size %d\n", size);
	(*benchmark) = (char *)calloc((size),sizeof(char));
        memcpy((*benchmark), line, size-1);


        size = getline(&line, &len, image_file);
	(*output) = (char *)calloc((size - 1),sizeof(char));
        memcpy((*output), line, size-1);

        size = getline(&line, &len, image_file);
	(*ext) = (char *)calloc((size-1),sizeof(char));
	printf("size = %d\n", size);
        memcpy((*ext), line, size-1);

        getline(&line, &len, image_file);
	int temp = atoi(line);
        memcpy(first_frame, &temp, sizeof(int));

        getline(&line, &len, image_file);
	temp = atoi(line);
        memcpy(last_frame, &temp, sizeof(int));

        return 0;
}


void serial_object_tracker(float *base_histogram, float **histogram, float *kernel, short *bins, short *object_w, short *object_h, int max_size, int max_index, int object_index, char *dx, char *dy ) {

        int i;

        for(i=0; i < max_index; i++) {
                (*histogram)[object_index*max_index + i] = 0;
        }

        float weight_sum = 0;
        float x_sum = 0;
        float y_sum = 0;

        //calculate Pu

        for(i=0; i < object_w[object_index]*object_h[object_index]; i++) {
                (*histogram)[object_index*max_index + (int)bins[object_index*max_size + i]] += kernel[object_index*max_size + i];
        }
        for(i = 0; i < max_index; i ++) {
                if((*histogram)[object_index * max_index + i] != 0) {
                        (*histogram)[object_index * max_index + i] = sqrt(base_histogram[object_index * max_index + i]/(*histogram)[object_index * max_index + i]);
                }
        }

        float curPixelWeight;
        for (i = 0; i < object_w[object_index] * object_h[object_index]; i++) {
                if (kernel[object_index*max_size + i] != 0){
                        curPixelWeight = (*histogram)[object_index*max_index + (int)bins[object_index*max_size + i]];
                        weight_sum += curPixelWeight;
                        x_sum += (i%object_w[object_index] - (object_w[object_index] >> 1))*curPixelWeight;
                        y_sum += (i/object_w[object_index] - (object_h[object_index] >> 1))*curPixelWeight;
                }
        }

        if( weight_sum != 0) {
                if(x_sum/weight_sum > 0 && x_sum/weight_sum < 0.01)
                        (*dx) = floor(x_sum/weight_sum);
                else if(x_sum/weight_sum > 0.01 && x_sum/weight_sum < 1)
                        (*dx) = ceil(x_sum/weight_sum);
                else if(x_sum/weight_sum > 1)
                        (*dx) = rint(x_sum/weight_sum);
                else if(x_sum/weight_sum < 0 && x_sum/weight_sum > -0.01)
                        (*dx) = ceil(x_sum/weight_sum);
                else if(x_sum/weight_sum < -0.01 && x_sum/weight_sum > -1)
                        (*dx) = floor(x_sum/weight_sum);
                else if(x_sum/weight_sum < -1)
                        (*dx) = rint(x_sum/weight_sum);

                if(y_sum/weight_sum > 0 && y_sum/weight_sum < 0.01)
                        (*dy) = floor(y_sum/weight_sum);
                else if(y_sum/weight_sum > 0.01 && y_sum/weight_sum < 1)
                        (*dy) = ceil(y_sum/weight_sum);
                else if(y_sum/weight_sum > 1)
                        (*dy) = rint(y_sum/weight_sum);
                else if(y_sum/weight_sum < 0 && y_sum/weight_sum > -0.01)
                        (*dy) = ceil(y_sum/weight_sum);
                else if(y_sum/weight_sum < -0.01 && y_sum/weight_sum > -1)
                        (*dy) = floor(y_sum/weight_sum);
                else if(y_sum/weight_sum < -1)
                        (*dy) = rint(y_sum/weight_sum);
        }
        else
        {
                (*dx) = 0;
                (*dy) = 0;
        }
}



void object_track(int num_objects, short *object_x, short *object_y, short *object_w, short *object_h, int first_frame, int last_frame, char *base_address, char *output_folder, char *ext, char device_type, int local_size, int search_distance, int threshold, int bin_resolution, char par, char lut) {

	cl_mem base_histogram_d;
	cl_mem kernel_d, bins_d;
	cl_mem object_w_d, object_h_d;
	cl_mem dx_d, dy_d;
	cl_mem weight_sum_d, x_sum_d, y_sum_d;
	cl_mem max_object_size_d;
	cl_mem max_index_d;
	cl_mem histogram_d;
	cl_mem distance_d;
	cl_mem converged_objects_d;
	char num_neighbors;

	int bin_size;
	int bins_per_color;
	int num_bins;
	short max_LUT_index;

	int width;
	int height;

	int loop_threshold;

        FILE *f;
        f = fopen("../Benchmarks/Benchmark4/high-quality/quality.txt", "a");

	cl_int ret;
	cl_platform_id platform;
	cl_uint num_platforms;
	//cl_init = 0;
	cl_device_id device;
	cl_uint num_devices;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel clkernel;
	cl_event event;
	cl_int status;

        int max_index;
        //timeing variables

        float computation_runtime = 0;
        float total_runtime = 0;
        struct timeval computation_start, computation_finish;
        struct timeval total_start, total_finish;

        //NDRange variables
        size_t global_work_size;
        size_t local_work_size;

        int max_object_size = 0;

        //histogram variables
        float *base_histogram;
        float *histogram;
        short *bins;
        float *kernel;
        unsigned char *rgb_frame;
        int *frame_indexes;
        short *lookup_table;

        //shift vectors
        char *dx;
        char *dy;
        float *x_sum;
        float *y_sum;
        float *weight_sum;
        char *last_dx;
        char *last_dy;
        char *converged_objects;
        int num_converged_objects;
	short *frame_pos;
        //benchmark
        char benchmark[200];

        static int mem_init = 0;
        int frameNumber = 0;

        int *loopCount;
        char exit_loop = 0;

        int kernelCount = 0;
	char *cl_file;

        int i;
	        cl_file = "coarse_lut_fine_parallel_tracker.cl";
	if(local_size > 0 && search_distance > 0 && lut == 1)
	        cl_file = "coarse_lut_fine_parallel_tracker.cl";
        num_neighbors = pow((search_distance*2 + 1), 2);

	bin_size = bin_resolution;
	bins_per_color = 256 / bin_size;
	num_bins = pow(bins_per_color, 3);

        max_LUT_index = 0;

        width = 0;
        height = 0;

	if(par == 1) {



                status = clGetPlatformIDs(1, &platform, NULL);
                if(status != CL_SUCCESS) {printf("reason: %s\n", status); printf("clGetPlatformIDs failed\n");exit(-1);}

                char buf[100];
                status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR,
                        sizeof(buf), buf, NULL);
                //  printf("\tVendor: %s\n", buf);
                status |= clGetPlatformInfo(platform, CL_PLATFORM_NAME,
                        sizeof(buf), buf, NULL);
                //  printf("\tName: %s\n", buf);

                if(status != CL_SUCCESS) {
                        printf("clGetPlatformInfo failed\n");
                        exit(-1);
                }
//              cl_device_id *device;
                // Retrieve the number of devices present
		if(device_type == 0)
                	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
		else if(device_type == 1)
                	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

                if(status != CL_SUCCESS) {printf("reason: %s\n", status); printf("clGetDeviceIDs failed\n");exit(-1);}

                status = clGetDeviceInfo(device, CL_DEVICE_VENDOR,
                        sizeof(buf), buf, NULL);
                printf("\tDevice: %s\n", buf);
                status |= clGetDeviceInfo(device, CL_DEVICE_NAME,
                        sizeof(buf), buf, NULL);
                printf("\tName: %s\n", buf);

                cl_ulong mem_size;
                status = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE , sizeof(cl_ulong), &mem_size, NULL);
                //chk(status);
                //  printf("Device Local Memory Size: %d KB\n", mem_size/1024);


                if(status != CL_SUCCESS) {
                        printf("clGetDeviceInfo failed\n");
                        exit(-1);
                }
                context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
                if(status != CL_SUCCESS) {printf("reason: %s\n", status); printf("clCreateContext failed\n");exit(-1);}
                command_queue = clCreateCommandQueue(context, device,
                CL_QUEUE_PROFILING_ENABLE, &status);
                if(status != CL_SUCCESS) {printf("reason: %s\n", status); printf("clCreateCommandQueue\n");exit(-1);}

                //set up GPU code

                char *source_code;
                size_t source_length;

                //read from source file
                source_code = read_buffer(cl_file, &source_length);

                //create a program

                program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_length, &ret);
                if(ret != CL_SUCCESS)
                {
                  printf("error: call to 'clCreateProgramWithSource' failed\n");
                  exit(1);
                }

                printf("program=%p\n", program);
                printf("\n");

                 //build program
                //ret = clBuildProgram(program, 1, &device, "-Werror -cl-mad-enable", NULL, NULL);
                //ret = clBuildProgram(program, 1, &device, "-I ./ -g -O0", NULL, NULL);
                ret = clBuildProgram(program, 1, &device, "-Werror -cl-mad-enable", NULL, NULL);
                if(ret != CL_SUCCESS)
                {
                	size_t size;
                	char *log;
                  	//get log size
                  	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
                  	//alloc log and print
                  	log = malloc(size);
                  	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
                  	printf("error: call to 'clBuildProgram' failed:\n%s\n", log);

                  	//feree log and exit
                  	free(log);
                  	exit(1);
                }
                printf("progam built\n");
                printf("\n");

                 //create a kernel
                clkernel = clCreateKernel(program, "track_object", &ret);
                if(ret != CL_SUCCESS)
                {
                  printf("error: call to 'clCreateKernel' failed\n");
                  exit(1);
                }

                printf("kernel=%p\n", clkernel);
                printf("\n");

	}

        loopCount = (int *)calloc(num_objects, sizeof(int));
        dx = (char *)calloc(num_neighbors * num_objects,sizeof(char));
        dy = (char *)calloc(num_neighbors * num_objects,sizeof(char));
        weight_sum = (float *)calloc(3*num_objects*num_neighbors*local_size, sizeof(float));
        last_dx = (char *)calloc(num_neighbors * num_objects,sizeof(char));
        last_dy = (char *)calloc(num_neighbors * num_objects,sizeof(char));
        converged_objects = (char *)calloc(num_objects,sizeof(char));
        frame_indexes = (int *)calloc(num_objects,sizeof(int));
	frame_pos = (short *)calloc((last_frame - first_frame + 1)*num_objects*2, sizeof(short));
        local_work_size = local_size;
        global_work_size = local_size*num_objects*num_neighbors;// max with numbins????

        for (i = first_frame; i <= last_frame; i++) {
                sprintf(benchmark, "%s%d.%s", base_address, i, ext);		
                decode_image(benchmark, &rgb_frame, ((last_frame - first_frame) + 1), i - first_frame, &width, &height);
        }

        for (i = 0; i < 3*num_objects*num_neighbors*local_size; i++) {
                weight_sum[i] = 0;
        }
        int max_size = 0;

        for (i = 0; i < num_objects; i++) {
                if (max_size < (object_w[i]+2*search_distance)*(object_h[i]+2*search_distance))
                        max_size = (object_w[i]+2*search_distance)*(object_h[i]+2*search_distance);
        }


	kernel = (float *)malloc(num_objects * max_size * sizeof(float));
        bins = (short *)malloc(num_objects * max_size * sizeof(short));	
        lookup_table = (short *)calloc(num_bins, sizeof(short));
	max_index = num_bins;
        Kernel_calculate(&kernel, 0, num_objects, max_size, object_w, object_h);
	if (lut == 1) {
	        clearLookupTable(&lookup_table, &max_index, num_bins);

	        updateLookupTable(rgb_frame, &lookup_table, &max_index, object_x, object_y, object_w, object_h, num_objects, width, bin_size, bins_per_color);
	}
        histogram = (float *)calloc(num_objects * max_index, sizeof(float));
        base_histogram = (float *)calloc(num_objects * max_index , sizeof(float));

        printf("size = %d\n", local_size*num_objects*num_neighbors*max_index);
	

	if(lut == 1)
        	update_bins_lut(rgb_frame, &bins, lookup_table, 0, num_objects, max_size, object_x, object_y, object_w, object_h, converged_objects, frame_indexes, num_neighbors, width, height, bin_size, bins_per_color);
	else
		update_bins(rgb_frame, &bins, 0, num_objects, max_size, object_x, object_y, object_w, object_h, converged_objects, frame_indexes, num_neighbors, width, height, bin_size, bins_per_color);
	
        for (i = 0; i < num_objects; i++) {
                frame_indexes[i] ++;
		frame_pos[i*2] = object_x[i];
		frame_pos[i*2 + 1] = object_y[i];
        }

        updateQ(&base_histogram, bins, kernel, max_index, 0, num_objects, max_size, object_x, object_y, object_w, object_h, num_neighbors);

	
	if (par == 1) {

	        base_histogram_d = clCreateBuffer(context, CL_MEM_READ_ONLY, num_objects*max_index*sizeof(float), NULL, &ret);
        	histogram_d = clCreateBuffer(context, CL_MEM_READ_WRITE, local_size*num_neighbors*num_objects*max_index*sizeof(float), NULL, &ret);
	        kernel_d = clCreateBuffer(context, CL_MEM_READ_ONLY, num_objects * max_size*sizeof(float), NULL, &ret);
        	bins_d = clCreateBuffer(context, CL_MEM_READ_ONLY, num_objects * max_size*sizeof(short), NULL, &ret);
	        object_w_d = clCreateBuffer(context, CL_MEM_READ_ONLY, num_objects*sizeof(short), NULL, &ret);
        	object_h_d = clCreateBuffer(context, CL_MEM_READ_ONLY, num_objects*sizeof(short), NULL, &ret);
	        max_index_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short), NULL, &ret);
        	dx_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_objects*num_neighbors*sizeof(char), NULL, &ret);
	        dy_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_objects*num_neighbors*sizeof(char), NULL, &ret);
        	max_object_size_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
	        distance_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
        	converged_objects_d = clCreateBuffer(context, CL_MEM_READ_ONLY, num_objects*sizeof(char), NULL, &ret);
	//        x_sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_objects*num_neighbors*local_size*sizeof(float), NULL, &ret);
	//        y_sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_objects*num_neighbors*local_size*sizeof(float), NULL, &ret);
        	weight_sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*num_objects*num_neighbors*local_size*sizeof(cl_float), NULL, &ret);



	        ret = clEnqueueWriteBuffer(command_queue, base_histogram_d, CL_TRUE, 0, num_objects*max_index*sizeof(float), base_histogram, 0, NULL, NULL);
        	ret = clEnqueueWriteBuffer(command_queue, max_index_d, CL_TRUE, 0, sizeof(short), &max_index, 0, NULL, NULL);
	        ret = clEnqueueWriteBuffer(command_queue, kernel_d, CL_TRUE, 0, num_objects*max_size*sizeof(float), kernel, 0, NULL, NULL);
        	ret = clEnqueueWriteBuffer(command_queue, max_object_size_d, CL_TRUE, 0, sizeof(int), &max_size, 0, NULL, NULL);
	        ret = clEnqueueWriteBuffer(command_queue, object_w_d, CL_TRUE, 0, num_objects*sizeof(short), object_w, 0, NULL, NULL);
        	ret = clEnqueueWriteBuffer(command_queue, object_h_d, CL_TRUE, 0, num_objects*sizeof(short), object_h, 0, NULL, NULL);
	        ret = clEnqueueWriteBuffer(command_queue, distance_d, CL_TRUE, 0, sizeof(int), &search_distance, 0, NULL, NULL);
        	ret = clEnqueueWriteBuffer(command_queue, converged_objects_d, CL_TRUE, 0, num_objects*sizeof(char), converged_objects, 0, NULL, NULL);


	}

        gettimeofday(&total_start, NULL);

        char finished = 0;
        num_converged_objects = 0;
	if(par == 1) {
        while (finished == 0) {


//                clear_buffers(&dx, &dy, &converged_objects);

		if(lut == 1)
        		update_bins_lut(rgb_frame, &bins, lookup_table, 0, num_objects, max_size, object_x, object_y, object_w, object_h, converged_objects, frame_indexes, num_neighbors, width, height, bin_size, bins_per_color);
		else
			update_bins(rgb_frame, &bins, 0, num_objects, max_size, object_x, object_y, object_w, object_h, converged_objects, frame_indexes, num_neighbors, width, height, bin_size, bins_per_color);


                ret = clEnqueueWriteBuffer(command_queue, bins_d, CL_TRUE, 0, num_objects*max_size*sizeof(short), bins, 0, NULL, NULL);
                        // set arguments for the kernel //

                ret = clSetKernelArg(clkernel, 0, sizeof(cl_mem), (void*)&base_histogram_d);
//                ret = clSetKernelArg(clkernel, 1, max_index*sizeof(float), NULL);
                ret = clSetKernelArg(clkernel, 1, sizeof(cl_mem), (void*)&histogram_d);
                ret = clSetKernelArg(clkernel, 2, sizeof(cl_mem), (void*)&kernel_d);
                ret = clSetKernelArg(clkernel, 3, sizeof(cl_mem), (void*)&bins_d);
                ret = clSetKernelArg(clkernel, 4, sizeof(cl_mem), (void*)&max_index_d);
                ret = clSetKernelArg(clkernel, 5, sizeof(cl_mem), &weight_sum_d);
//                ret = clSetKernelArg(clkernel, 6, sizeof(cl_mem), &x_sum_d);
//                ret = clSetKernelArg(clkernel, 7, sizeof(cl_mem), &y_sum_d);
                ret = clSetKernelArg(clkernel, 6, sizeof(cl_mem), (void*)&object_w_d);
                ret = clSetKernelArg(clkernel, 7, sizeof(cl_mem), (void*)&object_h_d);
                ret = clSetKernelArg(clkernel, 8, sizeof(cl_mem), (void*)&max_object_size_d);
                ret = clSetKernelArg(clkernel, 9, sizeof(cl_mem), (void*)&distance_d);
                ret = clSetKernelArg(clkernel, 10, sizeof(cl_mem), (void*)&converged_objects_d);

                if(ret != CL_SUCCESS)
                {
                        printf("error: call to 'clSetKernelArg' failed \n");
                }

                gettimeofday(&computation_start, NULL);
                ret = clEnqueueNDRangeKernel(command_queue, clkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
//                clWaitForEvents(1, &event);
                clFinish(command_queue);
                gettimeofday(&computation_finish, NULL);
                computation_runtime += ((float) (computation_finish.tv_usec - computation_start.tv_usec)/1000000 + (float) (computation_finish.tv_sec - computation_start.tv_sec));
                kernelCount ++;
                if(ret != CL_SUCCESS)
                {
                        printf("error: call to 'clEnqueueNDRangeKernel' failed \n");
                }
                ret = clEnqueueReadBuffer(command_queue, weight_sum_d, CL_TRUE, 0, sizeof(cl_float)*3*num_objects*num_neighbors*local_size, weight_sum, 0, NULL, &event);
                clWaitForEvents(1, &event);
                int j;
                for (i = 0; i < num_objects*num_neighbors; i++) {
                        for (j = 1; j < local_size; j++) {
                                weight_sum[3*i*local_size] += weight_sum[3*(i*local_size+j)];
                                weight_sum[3*i*local_size+1] += weight_sum[3*(i*local_size+j)+1];
                                weight_sum[3*i*local_size+2] += weight_sum[3*(i*local_size+j)+2];
                        }
                }

                for (i = 0; i < num_objects*num_neighbors; i++) {
                        if(weight_sum[3*i*local_size] != 0) {
                                float x_temp = weight_sum[3*i*local_size+1]/weight_sum[3*i*local_size];
                                float y_temp = weight_sum[3*i*local_size+2]/weight_sum[3*i*local_size];
                                if(x_temp > 0 && x_temp < 0.01)
                                        dx[i] = floor(x_temp);
                                else if(x_temp >= 0.01 && x_temp < 1)
                                        dx[i] = ceil(x_temp);
                                else if(x_temp >= 1 )
                                        dx[i] = rint(x_temp);
                                else if(x_temp <0 && x_temp > -0.01)
                                        dx[i] = ceil(x_temp);
                                else if(x_temp <= -0.01 && x_temp > -1)
                                        dx[i] = floor(x_temp);
                                else if(x_temp <= -1)
                                        dx[i] = rint(x_temp);
                                if(y_temp > 0 && y_temp < 0.01)
                                        dy[i] = floor(y_temp);
                                else if(y_temp >= 0.01 && y_temp < 1)
                                        dy[i] = ceil(y_temp);
                                else if(y_temp >= 1 )
                                        dy[i] = rint(y_temp);
                                else if(y_temp <0 && y_temp > -0.01)
                                        dy[i] = ceil(y_temp);
                                else if(y_temp <= -0.01 && y_temp > -1)
                                        dy[i] = floor(y_temp);
                                else if(y_temp <= -1)
                                        dy[i] = rint(y_temp);
                        }
                        else {
                                dx[i] = 0;
                                dy[i] = 0;
                        }
                }
                int last_converged_count = num_converged_objects;
                update_coordinate(dx, dy, &last_dx, &last_dy, &object_x, &object_y, &converged_objects, &num_converged_objects, &loopCount, &frame_indexes, last_frame - first_frame + 1, num_objects, num_neighbors, width, height, threshold, &frame_pos);
                if(last_converged_count != num_converged_objects)

                        ret = clEnqueueWriteBuffer(command_queue, converged_objects_d, CL_TRUE, 0, num_objects*sizeof(char), converged_objects, 0, NULL, NULL);
                if(num_converged_objects >= num_objects)
                        finished = 1;


//                writeFrame(&rgb_frame, output_address, ext, frameNumber-start_frame-1, object_x, object_y, object_w, object_h);

        }
	}

	else { //Serial
	int frameNumber;
	int object_index;

	for(frameNumber = 1; frameNumber <= last_frame - first_frame + 1; frameNumber++)
        {
                char str[100];
        //        sprintf(str, "%s%d.%s",base_address, frameNumber, ext);
        //        decode_image(str, &rgb_frame);
			update_bins(rgb_frame, &bins, 0, num_objects, max_size, object_x, object_y, object_w, object_h, converged_objects, frame_indexes, 1, width, height, bin_size, bins_per_color);

               
		for (object_index = 0; object_index < num_objects; object_index ++) {


                        dx[object_index] = 0;
                        dy[object_index] = 0;
                        last_dx[object_index] = 0;
                        last_dy[object_index] = 0;
                        exit_loop = 0;
                        loopCount[object_index] = 0;
                        while(exit_loop == 0)
                        {
                                loopCount[object_index] ++;
                                last_dx[object_index] = dx[object_index];
                                last_dy[object_index] = dy[object_index];


                                gettimeofday(&computation_start, NULL);
                                serial_object_tracker(base_histogram, &histogram, kernel, bins, object_w, object_h, max_size, num_bins, object_index, &dx[object_index], &dy[object_index]);
                                gettimeofday(&computation_finish, NULL);


                		kernelCount ++;
                                computation_runtime += ((float) (computation_finish.tv_usec - computation_start.tv_usec)/1000000 + (float) (computation_finish.tv_sec - computation_start.tv_sec));
                                //printf("dx = %d, dy = %d\n", dx, dy);
                            	if( (last_dx[object_index] + dx[object_index] == 0) && (last_dy[object_index] + dy[object_index] == 0)) {
                                        if(dx[object_index] >= 0)
                                                dx[object_index] += (threshold - loopCount[object_index])/3;
                                        else if(dx[object_index] < 0)
                                                dx[object_index] -= (threshold - loopCount[object_index])/3;
                                        if(dy[object_index] >= 0)
                                                dy[object_index] += (threshold - loopCount[object_index])/3;
                                        else if(dy[object_index] < 0)
                                                dy[object_index] -= (threshold - loopCount[object_index])/3;
//                                      dx *= 10;
//                                      dy *= 10;
                                }



				object_x[object_index] += dx[object_index];
                                object_y[object_index] += dy[object_index];
                                if(object_x[object_index] < 0)
                                        object_x[object_index] = 0;
                                if(object_y[object_index] < 0)
                                        object_y[object_index] = 0;

                                if(object_x[object_index] > width)
                                       object_x[object_index] = width;
                                if(object_y[object_index] > height)
                                        object_y[object_index] = height;

                        if( ( (dx[object_index] == 0) && (dy[object_index] == 0)) || (loopCount[object_index] > threshold) || ( (last_dx[object_index] + dx[object_index] == 0) && (last_dy[object_index] + dy[object_index] == 0)) ) {
                                        exit_loop = 1;
					frame_indexes[object_index]++;
					frame_pos[frameNumber*num_objects*2 + object_index * 2] = object_x[object_index];
	        			frame_pos[frameNumber*num_objects*2 + object_index * 2 + 1] = object_y[object_index];

                                }

                                else
                                {

					update_bins(rgb_frame, &bins, object_index, object_index+1, max_size, object_x, object_y, object_w, object_h, converged_objects, frame_indexes, 1, width, height, bin_size, bins_per_color);
                                }
                        }


                }

        }


	}

        gettimeofday(&total_finish, NULL);
	writeFrames(&rgb_frame, output_folder, ext, frame_pos, object_w, object_h, num_objects, width, height, last_frame-first_frame+1);


        total_runtime = ((float) (total_finish.tv_usec - total_start.tv_usec)/1000000 + (float) (total_finish.tv_sec - total_start.tv_sec));
        printf("kernel time : %f\n", computation_runtime);
        printf("computation performance (FPS) = %f\n", (last_frame-first_frame)/computation_runtime);
        printf("total time : %f\n", total_runtime);
        printf("performance (FPS) = %f\n", (last_frame-first_frame)/total_runtime);
        printf("\nFinal positions:\n");
        for (i = 0; i < num_objects; i++) {
                printf("object %d: %d %d\n", i, object_x[i], object_y[i]);
        }
        printf("kernel count %d\n", kernelCount);

        free(kernel);
        free(bins);
        free(base_histogram);
//        free(histogram);
        free(rgb_frame);

        clReleaseKernel(clkernel);
        clReleaseProgram(program);
        clReleaseMemObject(bins_d);
        clReleaseMemObject(base_histogram_d);
//        clReleaseMemObject(histogram_d);
        clReleaseMemObject(dx_d);
        clReleaseMemObject(dy_d);
        clReleaseMemObject(kernel_d);
        clReleaseMemObject(object_w_d);
        clReleaseMemObject(object_h_d);


//	chan *converged_objects = (char *)calloc(num_objects, sizeof(char));

}

int main(int argc, char **argv) {


	char *object_file;
	char *image_file;
	char *benchmark;
	char *output_folder;
	char *ext;

	int num_objects;
	char device_type = 'g';

	
	short *object_x;
	short *object_y;
	short *object_w;
	short *object_h;
	
	int first_frame;
	int last_frame;

	int local_size = 1;
	int search_distance = 0;
	
	char par = 0;
	char fine_par = 1;
	char coarse_par = 1;
	char lut = 1;
	int threshold = 50;
	int bin_resolution = 16;
	
	int i = 1;

	while ( i < argc ) {
	
		if (!strcmp(argv[i], "-m")) {
			par = atoi(argv[++i]);
		}

		else if (!strcmp(argv[i], "-o")) {
			object_file = malloc((strlen(argv[++i]) - 1)*sizeof(char));
			strcpy(object_file, argv[i]);
		}

		else if (!strcmp(argv[i], "-i")) {
			image_file = malloc((strlen(argv[++i]) - 1)*sizeof(char));
			strcpy(image_file, argv[i]);
		}

		if (!strcmp(argv[i], "-f")) {
			local_size = atoi(argv[++i]);
		}

		if (!strcmp(argv[i], "-c")) {
			search_distance = atoi(argv[++i]);
		}

		if (!strcmp(argv[i], "-l")) {
			lut = atoi(argv[++i]);
		}

		if (!strcmp(argv[i], "-t")) {
			threshold = atoi(argv[++i]);
		}

		if (!strcmp(argv[i], "-d")) {
			device_type = atoi(argv[++i]);
		}

		if (!strcmp(argv[i], "-b")) {
			bin_resolution = atoi(argv[++i]);
		}

		i++;
	}

	read_object_file(object_file, &num_objects, &object_x, &object_y, &object_w, &object_h);

	read_image_file(image_file, &benchmark, &output_folder, &ext, &first_frame, &last_frame);
	ext = "jpg";
	printf("par = %d", par);
printf("bench %s\n", benchmark);
printf("out %s\n", output_folder);

printf("ext %s\n", ext);

	object_track(num_objects, object_x, object_y, object_w, object_h, first_frame, last_frame, benchmark, output_folder, ext, device_type, local_size, search_distance, threshold, bin_resolution, par, lut);

	return 0;
}
