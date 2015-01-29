

__kernel void track_object ( 	__global float *base_histogram,
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
				__global int *distance,
				__global char *converged_objects,
				__global char *dx,
				__global char *dy
								  ) 

{
	int i;
	int local_id = get_local_id(0);
	int global_id = get_global_id(0);
	int local_size = get_local_size(0);
	int num_neighbors = pow((float)(2*distance[0] + 1), 2);
	int object_index = (global_id) / num_neighbors;
	int neighbor_index = (global_id) % num_neighbors;

	if(converged_objects[object_index] == 0) {

	int maxIndex = max_index[0];

	for(i=0; i < maxIndex; i++) {
		histograms[i] = 0;
	}
	int maxSize = max_size[0];

	weight_sum[0] = 0;
	x_sum[0] = 0;
	y_sum[0] = 0;
	
	//calculate Pu
	int width = object_width[object_index];
	int height = object_height[object_index];
	int binIndex;
	int row;
	int col;
	int neighborRow;
	int neighborCol;


	for(i=0; i < width*height; i++) {
                row = i/width;
                col = i%width;
                neighborRow = neighbor_index / (2*distance[0] + 1);
                neighborCol = neighbor_index % (2*distance[0] + 1);
                binIndex = (row + neighborRow) * (width+2*distance[0]) + (col + neighborCol);
                histograms[(int)bins[object_index*maxSize + binIndex]] += kernelVal[object_index*maxSize + i];
	}

	int j;
	for(i = 0; i < maxIndex; i++) {
		if(histograms[i] != 0) {
			histograms[i] = sqrt(base_histogram[object_index * maxIndex + i]/histograms[i]);
		}
	}

	float curPixelWeight;
	for (i = 0; i < width * height; i++) {
		if (kernelVal[object_index*maxSize + i] != 0){
                        row = i/width;
                        col = i%width;
                	neighborRow = neighbor_index / (2*distance[0] + 1);
	                neighborCol = neighbor_index % (2*distance[0] + 1);
                        binIndex = (row + neighborRow) * (width+2*distance[0]) + (col + neighborCol);
                        curPixelWeight = histograms[(int)bins[object_index*maxSize + binIndex]];
                        weight_sum[0] += curPixelWeight;
                        x_sum[0] += (i%width - (width >> 1))*curPixelWeight;
                        y_sum[0] += (i/width - (height >> 1))*curPixelWeight;
		}
	}
	if( weight_sum[0] != 0) {
		dx[object_index*num_neighbors + neighbor_index] = floor(x_sum[0]/weight_sum[0]);
		dy[object_index*num_neighbors + neighbor_index] = floor(y_sum[0]/weight_sum[0]);
	}
	else 
	{
		dx[object_index*num_neighbors + neighbor_index] = 0;
		dy[object_index*num_neighbors + neighbor_index] = 0;
	}
	

	}
}
