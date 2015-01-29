
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>



int decode_image(char* filename, unsigned char **rgbframe, int numFrames, int index, int *width, int *height)
{
        IplImage *img = cvLoadImage(filename, -1);

        if((*width) == 0) {
                (*width) = img->width;
                (*height) = img->height;
                (*rgbframe) = (unsigned char *)malloc((numFrames+1) * (*width) * (*height) * 3 * sizeof(unsigned char));
		printf("num frames %d width %d height %d\n", numFrames, (*width), (*height));
        }
        int row, col;
        for( row = 0; row < (*height); row++ ) {
                for ( col = 0; col < (*width); col++ ) {
                        (*rgbframe)[(index * (*width) * (*height) * 3) + ((*width) * row + col) * 3] = img->imageData[((*width) * row + col) * 3];
                        (*rgbframe)[(index * (*width) * (*height) * 3) + ((*width) * row + col) * 3 + 1] = img->imageData[((*width) * row + col) * 3 + 1];
                        (*rgbframe)[(index * (*width) * (*height) * 3) + ((*width) * row + col) * 3 + 2] = img->imageData[((*width) * row + col) * 3 + 2];
                }
        }
        return 0;
}


void save_frame(unsigned char** rgbframe, char* base, char* ext, int idx, int width, int height)
{
        /* use OpenCV to write bmp file */
        static IplImage * pCVFile;
        int pixel_idx;
        char filename[100];

        pCVFile = cvCreateImage(cvSize(width, height), 8, 3);

        for(pixel_idx = 0; pixel_idx < width*height*3; pixel_idx ++)
        {
                pCVFile->imageData[pixel_idx] = (*rgbframe)[idx*width*height*3 + pixel_idx];
        }

        sprintf(filename, "%s/frame%d.%s", base, idx, ext);
        int p[3];

        p[0] = CV_IMWRITE_JPEG_QUALITY;
        p[1] = 95;
        p[2] = 0;
        cvSaveImage(filename, pCVFile, p);
        cvReleaseImage(&pCVFile);
}


void writeFrames(unsigned char** rgbframe, char* base, char* ext, short *frame_pos, short *object_width, short * object_height, int num_objects, int width, int height, int num_frames) {

	int f;
	for (f = 0; f < num_frames; f++) {
	        int i,j;
        	for (j = 0; j < num_objects; j++) {
                	for(i = frame_pos[f*num_objects*2 + j*2]; i < frame_pos[f*num_objects*2 + j*2] + object_width[j]; i++) {
                        	(*rgbframe)[f*width*height*3 + (frame_pos[f*num_objects*2 + j*2 + 1]*width + i) * 3] = 0;
	                        (*rgbframe)[f*width*height*3 + (frame_pos[f*num_objects*2 + j*2 + 1]*width + i) * 3 + 1] = 0;
        	                (*rgbframe)[f*width*height*3 + (frame_pos[f*num_objects*2 + j*2 + 1]*width + i) * 3 + 2] = 255;

                	        (*rgbframe)[f*width*height*3 + ((frame_pos[f*num_objects*2 + j*2 + 1]+object_height[j])*width + i) * 3] = 0;
                        	(*rgbframe)[f*width*height*3 + ((frame_pos[f*num_objects*2 + j*2 + 1]+object_height[j])*width + i) * 3 + 1] = 0;
	                        (*rgbframe)[f*width*height*3 + ((frame_pos[f*num_objects*2 + j*2 + 1]+object_height[j])*width + i) * 3 + 2] = 255;
        	        }

                	for(i = frame_pos[f*num_objects*2 + j*2 + 1]; i < frame_pos[f*num_objects*2 + j*2 + 1] + object_height[j]; i++) {
                        	(*rgbframe)[f*width*height*3 + (i*width + frame_pos[f*num_objects*2 + j*2]) * 3] = 0;
	                        (*rgbframe)[f*width*height*3 + (i*width + frame_pos[f*num_objects*2 + j*2]) * 3 + 1] = 0;
        	                (*rgbframe)[f*width*height*3 + (i*width + frame_pos[f*num_objects*2 + j*2]) * 3 + 2] = 255;

                	        (*rgbframe)[f*width*height*3 + (i*width + frame_pos[f*num_objects*2 + j*2]+object_width[j]) * 3] = 0;
                        	(*rgbframe)[f*width*height*3 + (i*width + frame_pos[f*num_objects*2 + j*2]+object_width[j]) * 3 + 1] = 0;
	                        (*rgbframe)[f*width*height*3 + (i*width + frame_pos[f*num_objects*2 + j*2]+object_width[j]) * 3 + 2] = 255;
        	        }
        	}
        	save_frame(rgbframe, base, ext, f, width, height);
	}

}

