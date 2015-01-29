

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
				__global char *converged_objects,
				__global char *dx,
				__global char *dy )
{
	int i;
	int object_index = get_global_id(0);
	if(converged_objects[object_index] == 0) {
	int maxIndex = max_index[0];

	for(i=0; i < maxIndex; i++) {
		histograms[object_index*maxIndex + i] = 0;
	}
	int maxSize = max_size[0];

	
	//calculate Pu
	int width = object_width[object_index];
	int height = object_height[object_index];
	
	for (i = 0; i < width*height; i++) {
		histograms[object_index * maxIndex + (int)bins[object_index * maxSize + i]] += kernelVal[object_index * maxSize + i];
	}

	for (i = 0; i < maxIndex; i++) {
		if(histograms[object_index * maxIndex + i] != 0) {
			histograms[object_index * maxIndex + i] = sqrt(base_histogram[object_index * maxIndex + i]/histograms[object_index * maxIndex + i]);
		}
	}

	weight_sum[0] = 0;
	x_sum[0] = 0;
	y_sum[0] = 0;
	float curPixelWeight;
	for (i = 0; i < width* height; i++) {
		if (kernelVal[object_index * maxSize + i] != 0) {
			curPixelWeight = histograms[object_index * maxIndex + (int)bins[object_index * maxSize + i]];
			weight_sum[0] += curPixelWeight;
			x_sum[0] += (i%width - (width >> 1))*curPixelWeight;
			y_sum[0] += (i/width - (height >> 1))*curPixelWeight;
		}
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

