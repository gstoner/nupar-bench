

__kernel void track_object ( 	__global float *base_histogram,
				__local float *base_histogram_local,
				__local float *histograms,
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
	int object_index = (global_id - local_id)/(local_size*9);
	int neighbor_index = ((global_id - local_id)/local_size) % 9;

	int maxIndex = max_index[0];

	for(i=0; i < maxIndex; i++) {
		histograms[local_id*maxIndex + i] = 0;
		base_histogram_local[i] = base_histogram[object_index*maxIndex + i];

	}
	int maxSize = max_size[0];
//	for(i=0; i < maxSize; i ++) {
//		kernelVal_local[i] = kernelVal[object_index*maxSize + i];
//		bins_local[i] = bins[object_index*maxSize + i];
//	}

	weight_sum[local_id] = 0;
	x_sum[local_id] = 0;
	y_sum[local_id] = 0;
	
	//calculate Pu
	int width = object_width[object_index];
	int height = object_height[object_index];
	int binIndex;
	int row;
	int col;
	int neighborRow;
	int neighborCol;

	for(i=local_id; i < width*height; i = i + local_size) {
		row = i/width;
		col = i%width;
		neighborRow = neighbor_index / 3;
		neighborCol = neighbor_index % 3;
		binIndex = (row + neighborRow) * (width+2) + (col + neighborCol);
		histograms[local_id*maxIndex + (int)bins[object_index*maxSize + binIndex]] += kernelVal[object_index*maxSize + i];
	}
	//barrier
	barrier(CLK_LOCAL_MEM_FENCE);
	//reduction
	int j;
	for(i = local_id; i < maxIndex; i = i + local_size) {
		for(j = 1; j < local_size; j++) {
			histograms[i] += histograms[j*maxIndex + i];
		}
		if(histograms[i] != 0) {
			histograms[i] = sqrt(base_histogram_local[i]/histograms[i]);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float curPixelWeight;
	for (i = local_id; i < width* height; i = i + local_size) {
		if (kernelVal[object_index*maxSize + i] != 0){
			row = i/width;
			col = i%width;
			neighborRow = neighbor_index / 3;
			neighborCol = neighbor_index % 3;
			binIndex = (row + neighborRow) * (width+2) + (col + neighborCol);
			curPixelWeight = histograms[(int)bins[object_index*maxSize + binIndex]];
			weight_sum[local_id] += curPixelWeight;
			x_sum[local_id] += (i%width - (width >> 1))*curPixelWeight;
			y_sum[local_id] += (i/width - (height >> 1))*curPixelWeight;
		}
	}
	//barrier
	barrier(CLK_LOCAL_MEM_FENCE);
	//reduction
	if(local_id == 0) {
		for(i=1; i<local_size; i++) {
			weight_sum[0] += weight_sum[i];
			x_sum[0] += x_sum[i];
			y_sum[0] += y_sum[i];
		}
		if( weight_sum[0] != 0) {
			dx[object_index*9 + neighbor_index] = floor(x_sum[0]/weight_sum[0]);
			dy[object_index*9 + neighbor_index] = floor(y_sum[0]/weight_sum[0]);
		}
		else 
		{
			dx[object_index*9 + neighbor_index] = 0;
			dy[object_index*9 + neighbor_index] = 0;
		}
	}
}
