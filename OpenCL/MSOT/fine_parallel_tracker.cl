

__kernel void track_object ( 	__global float *base_histogram,
				__global float *histograms,
				__global float *kernelVal,
				__global short *bins,
				__global short *max_index,
				__local float *weight_sum,
				__local float *x_sum,
				__local float *y_sum,
				__global short *object_width,
				__global short *object_height,
				__global short *max_size,
				__global char *dx,
				__global char *dy
								  ) 

{
	int i;
	int local_id = get_local_id(0);
	int global_id = get_global_id(0);
	int local_size = get_local_size(0);
	int object_index = (global_id - local_id)/(local_size);

	int maxIndex = max_index[0];

	for(i=0; i < maxIndex; i++) {
		histograms[global_id*maxIndex + i] = 0;
	}
	int maxSize = max_size[0];

	weight_sum[local_id] = 0;
	x_sum[local_id] = 0;
	y_sum[local_id] = 0;
	
	//calculate Pu
	int width = object_width[object_index];
	int height = object_height[object_index];

	for(i=local_id; i < width*height; i = i + local_size) {
		histograms[global_id*maxIndex + (int)bins[object_index*maxSize + i]] += kernelVal[object_index*maxSize + i];
	}
	//barrier
	barrier(CLK_GLOBAL_MEM_FENCE);
	//reduction
	int j;
	for(i = local_id; i < maxIndex; i = i + local_size) {
		for(j = 1; j < local_size; j++) {
			histograms[object_index * local_size * maxIndex + i] += histograms[object_index * local_size * maxIndex + j * maxIndex + i];
		}
		if(histograms[object_index * local_size * maxIndex + i] != 0) {
			histograms[object_index * local_size * maxIndex + i] = sqrt(base_histogram[object_index * maxIndex + i]/histograms[object_index * local_size * maxIndex + i]);
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	float curPixelWeight;
	for (i = local_id; i < width * height; i = i + local_size) {
		if (kernelVal[object_index*maxSize + i] != 0){
			curPixelWeight = histograms[object_index * local_size * maxIndex + (int)bins[object_index*maxSize + i]];
			weight_sum[local_id] += curPixelWeight;
			x_sum[local_id] += (i%width - (width >> 1))*curPixelWeight;
			y_sum[local_id] += (i/width - (height >> 1))*curPixelWeight;
		}
	}
	//barrier
	barrier(CLK_GLOBAL_MEM_FENCE);
	//reduction
	if(local_id == 0) {
		for(i=1; i<local_size; i++) {
			weight_sum[0] += weight_sum[i];
			x_sum[0] += x_sum[i];
			y_sum[0] += y_sum[i];
		}
		if( weight_sum[0] != 0) {
			dx[object_index] = floor(x_sum[0]/weight_sum[0]);
			dy[object_index] = floor(y_sum[0]/weight_sum[0]);
		}
		else 
		{
			dx[object_index] = 0;
			dy[object_index] = 0;
		}

	}
}
