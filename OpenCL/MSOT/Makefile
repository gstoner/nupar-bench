CC=gcc
INC_PATH=-I/usr/include \
	 -I/opt/AMDAPP/include \
	 -I.

LIB_PATH=-L/usr/lib/ \
	 -L/opt/AMDAPP/lib/x86

CFLAGS= -lopencv_core -lopencv_highgui -lopencv_imgproc -lOpenCL 
includes=$(whildcard ./*.h)

SRC_FILES=meanshift.c my_image.c my_meanshift.c

all: Meanshift

Meanshift: $(SRC_FILES) $(includes)
	$(CC) $(INC_PATH) $(LIB_PATH) -o $@ $(SRC_FILES) $(CFLAGS)

clean:
	rm -rf *.o Meanshift
