


int decode_image(char* filename, unsigned char **rgbframe, int numFrames, int index, int *width, int *height);

void save_frame(unsigned char** rgbframe, char* base, char* ext, int idx, int width, int height);

void writeFrames(unsigned char** rgbframe, char* base, char* ext, short *frame_pos, short *object_width, short * object_height, int num_objects, int width, int height, int num_frames);


