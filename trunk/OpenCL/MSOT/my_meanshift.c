
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>



// Finds the bin for the given pixel
unsigned int findBin(unsigned char R, unsigned char G, unsigned char B, int bin_size, int bins_per_color) {

        unsigned int r, g, b;
        r = (unsigned int)floor( (float)(R/bin_size));
        g = (unsigned int)floor( (float)(G/bin_size));
        b = (unsigned int)floor( (float)(B/bin_size));

        return (r + bins_per_color*g + bins_per_color*bins_per_color*b);
}


// Calculates kernel values
void Kernel_calculate(float** kernel, int first_index, int last_index, int max_object_size, short *width, short *height) {

        int i, j;
        int x,y;
        for(j = first_index; j < last_index; j++)
        {
                int mid_x = width[j]/2;
                int mid_y = height[j]/2;

                float distance;

                for (i = 0; i < height[j]*width[j]; i++) {
                        x = i % width[j];
                        y = i / width[j];

                        distance = pow(((float)(x-mid_x)/(float)(mid_x)), 2.0) + pow(((float)(y-mid_y)/(float)(mid_y)), 2.0);
                        if(distance >= 1) {
                                (*kernel)[j*max_object_size + i] = 0.0;
                        }
                        else {
                                (*kernel)[j*max_object_size + i] = 1 - distance;
                        }
                }
        }
}

void clearLookupTable(short **lookup_table, int *max_index, int num_bins) {

        int i;
        for(i = 0; i < num_bins; i++) {
                (*lookup_table)[i] = 0;
        }
        (*max_index) = 1;
}

int updateLookupTable(unsigned char *rgbframe, short **lookup_table, int *max_index, short *top_left_x, short *top_left_y, short *object_width, short *object_height, int num_objects, int width, int bin_size, int bins_per_color)
{
        int i;
        int row, col;
        for (i = 0; i < num_objects; i++) {
                for( row = 0; row < object_height[i]; row++ ) {
                        for ( col = 0; col < object_width[i]; col++ ) {
                                int index = findBin(rgbframe[(width * (row + top_left_y[i]) + (col + top_left_x[i])) * 3 + 2], rgbframe[(width * (row + top_left_y[i]) + (col + top_left_x[i])) * 3 + 1], rgbframe[(width * (row + top_left_y[i]) + (col + top_left_x[i])) * 3], bin_size, bins_per_color);
                                if((*lookup_table)[index] == 0) {
                                        (*lookup_table)[index] = (*max_index);
                                        (*max_index) ++;
                                }
                        }
                }
        }
        return 0;
}


int update_bins(unsigned char *rgbframe, short **bins, int first_index, int last_index, int max_object_size, short *top_left_x, short *top_left_y, short *object_width, short *object_height, char *converged_objects, int *frame_indexes, int num_neighbors, int width, int height, int bin_size, int bins_per_color)
{
        int i;
        int row, col;
        int search_distance = (sqrt(num_neighbors)-1)/2;

        for(i = first_index; i < last_index; i++) {
                if(converged_objects[i] == 0) {
                for(row = 0; row < object_height[i] + 2*search_distance; row++) {
                        for(col = 0; col < object_width[i] + 2*search_distance; col++) {
                                (*bins)[i*max_object_size + (object_width[i]+2*search_distance) * row + col] = findBin(rgbframe[frame_indexes[i]*width*height*3 + (width * (row-search_distance + top_left_y[i]) + (col-search_distance + top_left_x[i])) * 3 + 2], rgbframe[frame_indexes[i]*width*height*3 + (width * (row-search_distance + top_left_y[i]) + (col-search_distance + top_left_x[i])) * 3 + 1], rgbframe[frame_indexes[i]*width*height*3 + (width * (row-search_distance + top_left_y[i]) + (col-search_distance + top_left_x[i])) * 3], bin_size, bins_per_color);
                        }
                }
                }
        }

        return 0;
}


int update_bins_lut(unsigned char *rgbframe, short **bins, short *lookup_table, int first_index, int last_index, int max_object_size, short *top_left_x, short *top_left_y, short *object_width, short *object_height, char *converged_objects, int *frame_indexes, int num_neighbors, int width, int height, int bin_size, int bins_per_color)
{
        int i;
        int row, col;
        int search_distance = (sqrt(num_neighbors)-1)/2;
        for(i = first_index; i < last_index; i++) {
                if(converged_objects[i] == 0) {
                for(row = 0; row < object_height[i] + 2*search_distance; row++) {
                        for(col = 0; col < object_width[i] + 2*search_distance; col++) {
                                (*bins)[i*max_object_size + (object_width[i]+2*search_distance) * row + col] = lookup_table[findBin(rgbframe[frame_indexes[i]*width*height*3 + (width * (row-search_distance + top_left_y[i]) + (col-search_distance + top_left_x[i])) * 3 + 2], rgbframe[frame_indexes[i]*width*height*3 + (width * (row-search_distance + top_left_y[i]) + (col-search_distance + top_left_x[i])) * 3 + 1], rgbframe[frame_indexes[i]*width*height*3 + (width * (row-search_distance + top_left_y[i]) + (col-search_distance + top_left_x[i])) * 3], bin_size, bins_per_color)];
                        }
                }
                }

        }

        return 0;

}


void updateQ(float **Qu, short *bins, float *kernel, int max_index, int first_index, int last_index, int max_object_size, short *top_left_x, short *top_left_y, short *object_width, short *object_height, int num_neighbors) {

        int j;
        int x,y;
        int binIndex;
        int kernelIndex;
        int search_distance = (sqrt(num_neighbors)-1)/2;

                for (j = first_index; j < last_index; j++) {
                        for (y = 0; y < object_height[j]; y++) {
                                for (x = 0; x < object_width[j]; x++) {
                                        binIndex = (y+search_distance)*(object_width[j]+2*search_distance) + (x+search_distance);
                                        kernelIndex = (y)*object_width[j] + (x);
                                        (*Qu)[j*max_index + bins[j*max_object_size + binIndex]] += kernel[j*max_object_size + kernelIndex];
                                }
                        }


                        float total = 0;
                        int i;
                        for(i = 0; i < max_index; i++) {
                                total += (*Qu)[j*max_index + i];
                        }
                        for(i = 0; i < max_index; i++) {
                                (*Qu)[j*max_index + i] /= total;
                        }
                }

}


void clear_buffers(char **dx, char **dy, char **converged, int num_objects, int num_neighbors) {

        int i,j;
        for (i = 0; i < num_objects; i++) {
                for(j = 0; j < num_neighbors; j++) {
                        (*dx)[i*num_neighbors + j] = 0;
                        (*dy)[i*num_neighbors + j] = 0;
                }
                (*converged)[i] = 0;
        }
}


void next_frame(int index, int **loopCount, int **frame_indexes, char **converged_objects, int *num_converged_objects, int frame_count, char **pdx, char **pdy, int num_objects, short **frame_pos, short x, short y) {

	(*frame_pos)[(*frame_indexes)[index]*num_objects*2 + index * 2] = x;
	(*frame_pos)[(*frame_indexes)[index]*num_objects*2 + index * 2 + 1] = y;

        (*frame_indexes)[index] ++;
        (*loopCount)[index] = 0;
        (*pdx)[index] = 0;
        (*pdy)[index] = 0;
//      printf("object %d frame %d\n", index, (*frame_indexes)[index]);
        if((*frame_indexes)[index] > frame_count) {
                (*num_converged_objects) ++;
                (*converged_objects)[index] = 1;
                printf("object %d is done\n", index);

        }

}


void update_coordinate(char *dx, char *dy, char **pdx, char **pdy, short **object_x, short **object_y, char **converged_objects, int *num_converged_objects, int **loopCount, int **frame_indexes, int frame_count, int num_objects, int num_neighbors, int width, int height, int loop_threshold, short **frame_pos) {

        int i;
        int search_distance = (sqrt(num_neighbors)-1 ) / 2;

        for(i = 0; i < num_objects; i++) {

                if((*converged_objects)[i] == 0){
                int steps = 0;
                int neighbor_index = num_neighbors / 2;
                int movedx = 0;
                int movedy = 0;
                while (1) {
                                if( ((*pdx)[i] + dx[i*num_neighbors + neighbor_index] == 0) && ((*pdy)[i] + dy[i*num_neighbors + neighbor_index] == 0)) {
        
					if(dx[i*num_neighbors + neighbor_index] >= 0)
                                                dx[i*num_neighbors + neighbor_index] += (loop_threshold - (*loopCount)[i])/3;
                                        else if(dx[i*num_neighbors + neighbor_index] < 0)
                                                dx[i*num_neighbors + neighbor_index] -= (loop_threshold - (*loopCount)[i])/3;
                                        if(dy[i*num_neighbors + neighbor_index] >= 0)
                                                dy[i*num_neighbors + neighbor_index] += (loop_threshold - (*loopCount)[i])/3;
                                        else if(dy[i*num_neighbors + neighbor_index] < 0)
                                                dy[i*num_neighbors + neighbor_index] -= (loop_threshold - (*loopCount)[i])/3;

//	                                dx[i*num_neighbors + neighbor_index] *= 3;
//                                        dy[i*num_neighbors + neighbor_index] *= 3;
                                }

                        (*object_x)[i] += dx[i*num_neighbors + neighbor_index];
                        (*object_y)[i] += dy[i*num_neighbors + neighbor_index];
                        movedx += dx[i*num_neighbors + neighbor_index];
                        movedy += dy[i*num_neighbors + neighbor_index];

                        if((*object_x)[i] < 0)
                                (*object_x)[i] = 0;
                        if((*object_y)[i] < 0)
                                (*object_y)[i] = 0;

                        if((*object_x)[i] > width)
                                (*object_x)[i] = width;
                        if((*object_y)[i] > height)
                                (*object_y)[i] = height;

                        (*pdx)[i] = dx[i*num_neighbors+neighbor_index];
                        (*pdy)[i] = dy[i*num_neighbors+neighbor_index];

                                (*loopCount)[i] ++;
                        if(  ( (dx[i*num_neighbors+neighbor_index] == 0) && (dy[i*num_neighbors+neighbor_index] == 0)) || ((*loopCount)[i] > loop_threshold) || (((*pdx)[i] + dx[i*num_neighbors+neighbor_index] == 0) && ((*pdy)[i] + dy[i*num_neighbors+neighbor_index] == 0))) {
//                      if(  ( (dx[i*num_neighbors+neighbor_index] == 0) && (dy[i*num_neighbors+neighbor_index] == 0)) || ((*loopCount)[i] > loop_threshold) ) {
//                              next_frame()
                                next_frame(i, loopCount, frame_indexes, converged_objects, num_converged_objects, frame_count, pdx, pdy, num_objects, frame_pos, (*object_x)[i], (*object_y)[i]);
                                //write frame??????????????

                                break;
                        }
                        else if(abs(movedx) <= search_distance && abs(movedy) <=search_distance) {
                                neighbor_index += sqrt(num_neighbors)*dy[i*num_neighbors+neighbor_index] + dx[i*num_neighbors+neighbor_index];
        //                      (*pdx)[i] = dx[i*num_neighbors+neighbor_index];
        //                      (*pdy)[i] = dy[i*num_neighbors+neighbor_index];
                        }
                        else{
          //                    (*pdx)[i] = dx[i*num_neighbors+neighbor_index];
            //                          (*pdy)[i] = dy[i*num_neighbors+neighbor_index];
                                break;
                        }
                }
                }

        }

}




