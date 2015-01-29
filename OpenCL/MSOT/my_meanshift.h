

// Finds the bin for the given pixel
unsigned int findBin(unsigned char R, unsigned char G, unsigned char B, int bin_size, int bins_per_color);


// Calculates kernel values
void Kernel_calculate(float** kernel, int first_index, int last_index, int max_object_size, short *width, short *height);

void clearLookupTable(short **lookup_table, int *max_index, int num_bins);


int updateLookupTable(unsigned char *rgbframe, short **lookup_table, int *max_index, short *top_left_x, short *top_left_y, short *object_width, short *object_height, int num_objects, int width, int bin_size, int bins_per_color);


int update_bins(unsigned char *rgbframe, short **bins, int first_index, int last_index, int max_object_size, short *top_left_x, short *top_left_y, short *object_width, short *object_height, char *converged_objects, int *frame_indexes, int num_neighbors, int width, int height, int bin_size, int bins_per_color);


int update_bins_lut(unsigned char *rgbframe, short **bins, short *lookup_table, int first_index, int last_index, int max_object_size, short *top_left_x, short *top_left_y, short *object_width, short *object_height, char *converged_objects, int *frame_indexes, int num_neighbors, int width, int height, int bin_size, int bins_per_color);


void updateQ(float **Qu, short *bins, float *kernel, int max_index, int first_index, int last_index, int max_object_size, short *top_left_x, short *top_left_y, short *object_width, short *object_height, int num_neighbors);

void clear_buffers(char **dx, char **dy, char **converged, int num_objects, int num_neighbors);

void next_frame(int index, int **loopCount, int **frame_indexes, char **converged_objects, int *num_converged_objects, int frame_count, char **pdx, char **pdy, int num_objects, short **frame_pos, short x, short y);


void update_coordinate(char *dx, char *dy, char **pdx, char **pdy, short **object_x, short **object_y, char **converged_objects, int *num_converged_objects, int **loopCount, int **frame_indexes, int frame_count, int num_objects, int num_neighbors, int width, int height, int loop_threshold, short **frame_pos);




