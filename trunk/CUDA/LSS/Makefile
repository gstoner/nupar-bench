all: level_set_segmentation

level_set_segmentation:
	nvcc -arch=sm_35 -rdc=true level_set_segmentation.cu -o level_set_segmentation -lcudadevrt

clean:
	rm level_set_segmentation
