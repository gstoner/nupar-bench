
__kernel void track_object ( 	__global float *base_histogram,
				__global float *histograms,
				__global float *kernelVal,
				__global short *bins,
				__global short *max_index,
				__global float *weight_sum,
				__global short *object_width,
				__global short *object_height,
				__global short *max_size,
				__global int *distance,
				__global char *converged_objects) 

{

	int i;
	int local_id = get_local_id(0);
	int global_id = get_global_id(0);
	//int global_id = 0;
	int local_size = get_local_size(0);

	int d = distance[0];
	int num_neighbors = (2*d+1) * (2*d+1);
	int object_index = (global_id - local_id)/(local_size*num_neighbors);
	int neighbor_index = (global_id - local_id)/(local_size) % num_neighbors;
	if(converged_objects[object_index] == 0) {
	int maxIndex = max_index[0];
	for(i=0; i < maxIndex; i++) {
		histograms[global_id*maxIndex + i] = 0;
	}

	int maxSize = max_size[0];

	
	//calculate Pu
	int width = object_width[object_index];
	int height = object_height[object_index];
	int binIndex;
	int row;
	int col;
	int neighborRow;
	int neighborCol;

	weight_sum[3*global_id] = 0;
	weight_sum[3*global_id+1] = 0;
	weight_sum[3*global_id+2] = 0;

	for(i=local_id; i < width*height; i = i + local_size) {
                row = i/width;
                col = i - row*width;//i%width;
                neighborRow = neighbor_index / (2*d + 1);
                neighborCol = neighbor_index % (2*d + 1);
                binIndex = (row + neighborRow) * (width+2*d) + (col + neighborCol);
                histograms[global_id*maxIndex + (int)bins[object_index*maxSize + binIndex]] += kernelVal[object_index*maxSize + i];
	}
	//barrier
	barrier(CLK_GLOBAL_MEM_FENCE);
	//reduction
	int j;
	for(i = local_id; i < maxIndex; i = i + local_size) {
		for(j = 1; j < local_size; j++) {
			histograms[(global_id-local_id)*maxIndex + i] += histograms[(global_id-local_id)*maxIndex + j * maxIndex + i];
		}
		if(histograms[(global_id-local_id)*maxIndex + i] != 0) {
			histograms[(global_id-local_id)*maxIndex + i] = sqrt(base_histogram[object_index*maxIndex + i]/histograms[(global_id-local_id)*maxIndex + i]);
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	float curPixelWeight;
	for(i=local_id; i < width*height; i = i + local_size) {
		if (kernelVal[object_index*maxSize + i] != 0){
                        row = i/width;
                        col = i%width;
                	neighborRow = neighbor_index / (2*d + 1);
	                neighborCol = neighbor_index % (2*d + 1);
                        binIndex = (row + neighborRow) * (width+2*d) + (col + neighborCol);
                        curPixelWeight = histograms[(global_id-local_id) * maxIndex + (int)bins[object_index*maxSize + binIndex]];
                        weight_sum[3*global_id+0] += curPixelWeight;
                        weight_sum[3*global_id+1] += (col - (width >> 1))*curPixelWeight;
                        weight_sum[3*global_id+2] += (row - (height >> 1))*curPixelWeight;
		}
	}
	}

}
