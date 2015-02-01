#include <iostream>
#include <fstream>
#include <iomanip>
#include "image.h"

#define BUF_SIZE 256

using namespace std;

class errorPNM { };

struct Color
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

void readPNM(ifstream &file, char* buf);
image<unsigned char>* loadPGM(const char* name);
void savePPM(image<Color>* im, const char* name);
Color randomColor();

__global__ void evolveContour(unsigned char* intensityDev, unsigned char* labelsDev, signed char* speedDev, signed char* phiDev, int HEIGHT, int WIDTH, int* targetLabels, int kernelID, int numLabels, int* lowerIntensityBounds, int* upperIntensityBounds);

__global__ void initSpeedPhi(unsigned char* intensity, unsigned char* labels, signed char* speed, signed char* phi, int HEIGHT, int WIDTH, int targetLabel, int lowerIntensityBound, int upperIntensityBound);

__global__ void switchIn(signed char* speed, signed char* phi, int HEIGHT, int WIDTH);
__global__ void switchOut(signed char* speed, signed char* phi, int HEIGHT, int WIDTH);

__global__ void checkStopCondition(signed char* speed, signed char* phi, int parentThreadID, int HEIGHT, int WIDTH);
__device__ volatile int stopCondition[1024];


int main(int argc, char* argv[])
{
	// Parse command line arguments
	char* imageFile = NULL;
	char* labelFile = NULL;
	char* paramFile = NULL;
	int numRepetitions = 1;
	bool produceOutput = false;

	for(int i=1; i<argc; i++)
	{
		if(strcmp(argv[i], "--image") == 0)
		{
			if(i+1 < argc)
				imageFile = argv[++i];
			else
			{
				cerr << "Expected a filename after '" << argv[i] << "'. Try '" << argv[0]
					<< " --help' for additional information." << endl;
				exit(1);
			}
		}
		else if(strcmp(argv[i], "--labels") == 0)
		{
			if(i+1 < argc)
				labelFile = argv[++i];
			else
			{
				cerr << "Expected a filename after '" << argv[i] << "'. Try '" << argv[0]
					<< " --help' for additional information." << endl;
				exit(1);
			}
		}
		else if(strcmp(argv[i], "--params") == 0)
		{
			if(i+1 < argc)
				paramFile = argv[++i];
			else
			{
				cerr << "Expected a filename after '" << argv[i] << "'. Try '" << argv[0]
					<< " --help' for additional information." << endl;
				exit(1);
			}
		}
		else if(strcmp(argv[i], "--reps") == 0)
		{
			if(i+1 < argc)
			{
				numRepetitions = atoi(argv[++i]);
				if(numRepetitions < 1)
				{
					cerr << "Number of repetitions must be greater than 0." << endl;
					exit(1);
				}
			}
			else
			{
				cerr << "Expected a filename after '" << argv[i] << "'. Try '" << argv[0]
					<< " --help' for additional information." << endl;
				exit(1);
			}
		}
		else if(strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
			produceOutput = true;
		else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << "Usage: " << argv[0] << " [OPTIONS] --image <file> --labels <file> --params <file>" << endl;
			cout << "The order of switches does not matter so long as each one is immediately followed by its appropriate" << endl;
			cout << "argument (if one is required).\n" << endl;

			cout << "Utilizes a massively parallelized level set algorithm to segment the desired region of interest in" << endl;
			cout << "a grayscale image based on a given intensity range. Developed by Brett Daley as part of the NUPAR" << endl;
			cout << "benchmark suite.\n" << endl;

			cout << "Required arguments:" << endl;
			cout << "      --image <file>    Grayscale image to be segmented (intensities must be between 0 and 255)." << endl;
			cout << "      --labels <file>   Stripe-Based Connected Components Labeling output file. Used to seed the" << endl;
			cout << "                        initial contour." << endl;
			cout << "      --params <file>   Text file requiring the following format:" << endl;
			cout << "                           <Target label>" << endl;
			cout << "                           <Lower intensity bound>" << endl;
			cout << "                           <Upper intensity bound>" << endl;
			cout << "                           ..." << endl;
			cout << "                        Having multiple sets of three lines will segment the image multiple times according" << endl;
			cout << "                        to the different parameters. Utilizes dynamic parallelism." << endl;
			cout << "Options:" << endl;
			cout << "      --reps <number>   Run the program the specified number of times, enabling concurrent kernel execution" << endl;
			cout << "                        via Hyper-Q. Useful for performance benchmarking. [Default: 1]" << endl;
			cout << "  -o, --output          Output an RGB image for each target label specified in the params file. Use MATLAB's" << endl;
			cout << "                        'imshow' command to view." << endl;
			cout << "  -h, --help            Display this information and exit." << endl;

			exit(0);
		}
		else
		{
			cerr << "Did not recognize '" << argv[i] << "'. Try '" << argv[0]
				<< " --help' for additional information." << endl;
			exit(1);
		}
	}
	
	if(imageFile == NULL || labelFile == NULL || paramFile == NULL)
	{
		cerr << "Missing one or more arguments. Try '" << argv[0]
			<< " --help' for additional information." << endl;
		exit(1);
	}


        // Initialize timers, start the runtime timer
	cudaEvent_t startTime1, startTime2, stopTime1, stopTime2;
	cudaEventCreate(&startTime1);
	cudaEventCreate(&startTime2);
	cudaEventCreate(&stopTime1);
	cudaEventCreate(&stopTime2);
	float elapsedTime1, elapsedTime2;
	cudaEventRecord(startTime1, 0);


        // Load image, send to GPU
	image<unsigned char>* input = loadPGM(imageFile);
	const int HEIGHT = input->height();
	const int WIDTH = input->width();
	const int SIZE = HEIGHT*WIDTH*sizeof(char);

	unsigned char* intensity = new unsigned char[numRepetitions*HEIGHT*WIDTH];
	for(int i=0; i<numRepetitions; i++)
		memcpy(&intensity[i*HEIGHT*WIDTH], input->data, SIZE);

	unsigned char* intensityDev = NULL;
	cudaMalloc((void**)&intensityDev, numRepetitions*SIZE);
	cudaMemcpyAsync(intensityDev, intensity, numRepetitions*SIZE, cudaMemcpyHostToDevice);


        // Load connected component labels, send to GPU
	input = loadPGM(labelFile);

	unsigned char* labels = new unsigned char[numRepetitions*HEIGHT*WIDTH];
	for(int i=0; i<numRepetitions; i++)
		memcpy(&labels[i*HEIGHT*WIDTH], input->data, SIZE);

	unsigned char* labelsDev = NULL;
	cudaMalloc((void **)&labelsDev, numRepetitions*SIZE);
	cudaMemcpyAsync(labelsDev, labels, numRepetitions*SIZE, cudaMemcpyHostToDevice);


	// Load parameters, send to GPU
	ifstream paramStream;
	paramStream.open(paramFile);

	if(paramStream.is_open() != true)
	{
		cerr << "Could not open '" << paramFile << "'." << endl;
		exit(1);
	}

	int targetLabels[1024];
	int lowerIntensityBounds[1024];
	int upperIntensityBounds[1024];

	int numLabels = 0;
	while(paramStream.eof() == false)
	{
		char line[16];
		paramStream.getline(line, 16);
		
		if(paramStream.eof() == true)
			break;

		if(numLabels % 3 == 0)
			targetLabels[numLabels/3] = strtol(line, NULL, 10);
		else if(numLabels % 3 == 1)
			lowerIntensityBounds[numLabels/3] = strtol(line, NULL, 10);
		else
			upperIntensityBounds[numLabels/3] = strtol(line, NULL, 10);
		
		numLabels++;
	}
	
	if(numLabels % 3 == 0)
		numLabels /= 3;
	else
	{
		cerr << "Number of lines in " << paramFile << " is not divisible by 3. Try '" << argv[0]
			<< " --help' for additional information." << endl;
		exit(1);
	}
	paramStream.close();

	int* targetLabelsDev = NULL;
        cudaMalloc((void**)&targetLabelsDev, numLabels*sizeof(int));
        cudaMemcpyAsync(targetLabelsDev, targetLabels, numLabels*sizeof(int), cudaMemcpyHostToDevice);

        int* lowerIntensityBoundsDev = NULL;
        cudaMalloc((void**)&lowerIntensityBoundsDev, numLabels*sizeof(int));
        cudaMemcpyAsync(lowerIntensityBoundsDev, lowerIntensityBounds, numLabels*sizeof(int), cudaMemcpyHostToDevice);

        int* upperIntensityBoundsDev = NULL;
        cudaMalloc((void**)&upperIntensityBoundsDev, numLabels*sizeof(int));
        cudaMemcpyAsync(upperIntensityBoundsDev, upperIntensityBounds, numLabels*sizeof(int), cudaMemcpyHostToDevice);


        // Allocate arrays for speed and phi in GPU memory
	signed char* speedDev = NULL;
	signed char* phiDev = NULL;
	cudaMalloc((void**)&speedDev, numRepetitions*numLabels*SIZE);
	cudaMalloc((void**)&phiDev, numRepetitions*numLabels*SIZE);

	cudaDeviceSynchronize();


	// Start the segmentation timer
	cudaEventRecord(startTime2, 0);
	

	// Launch kernel to begin image segmenation
	for(int i=0; i<numRepetitions; i++)
	{
		evolveContour<<<1, numLabels>>>(intensityDev, labelsDev, speedDev, phiDev, HEIGHT, WIDTH, targetLabelsDev, i,
						numLabels, lowerIntensityBoundsDev, upperIntensityBoundsDev);
	}
	cudaDeviceSynchronize();


	// Stop the segmentation timer
	cudaEventRecord(stopTime2, 0);


	// Retrieve results from the GPU
	signed char* phi = new signed char[numRepetitions*numLabels*HEIGHT*WIDTH];
	cudaMemcpy(phi, phiDev, numRepetitions*numLabels*SIZE, cudaMemcpyDeviceToHost);


	// Stop the runtime timer
	cudaEventRecord(stopTime1, 0);


	// Output RGB images (if command line switch was present)
	if(produceOutput == true)
	{
		srand(time(NULL));
		Color colors[HEIGHT*WIDTH];
		for(int i=0; i<HEIGHT; i++)
			for(int j=0; j<WIDTH; j++)
				colors[i*WIDTH+j] = randomColor();

		for(int k=0; k<numLabels; k++)
		{
			image<Color> output = image<Color>(WIDTH, HEIGHT, true);
			image<Color>* im = &output;
			for(int i=0; i<HEIGHT; i++)
				for(int j=0; j<WIDTH; j++)
					im->access[i][j] = colors[phi[k*HEIGHT*WIDTH+i*WIDTH+j]];
			
			char filename[64];
			sprintf(filename, "segmented.target_label-%d.intensities-%d-%d.ppm", targetLabels[k], lowerIntensityBounds[k], upperIntensityBounds[k]);
			savePPM(im, filename);
		}
	}


        // Stop runtime timer and print times
	cudaEventElapsedTime(&elapsedTime1, startTime1, stopTime1);
	cudaEventElapsedTime(&elapsedTime2, startTime2, stopTime2);
	cout << "Computation time: " << setprecision(6) << elapsedTime2 << " ms"<< endl;
	cout << "Total time: " << setprecision(6) << elapsedTime1 << " ms"<< endl;
	

	// Free resources and end the program
	cudaEventDestroy(startTime1);
	cudaEventDestroy(stopTime1);
	cudaEventDestroy(startTime2);
	cudaEventDestroy(stopTime2);

	cudaFree(intensityDev);
	cudaFree(labelsDev);
	cudaFree(speedDev);
	cudaFree(phiDev);
	cudaFree(targetLabelsDev);
	cudaFree(lowerIntensityBoundsDev);
	cudaFree(upperIntensityBoundsDev);

        return 0;
}


image<unsigned char>* loadPGM(const char* name)
{
	char buf[BUF_SIZE];

	// Read header
	ifstream file(name, ios::in | ios::binary);
	readPNM(file, buf);
	if(strncmp(buf, "P5", 2))
	{
		cerr << "Unable to open '" << name << "'." << endl;
		throw errorPNM();
	}

	readPNM(file, buf);
	int width = atoi(buf);
	readPNM(file, buf);
	int height = atoi(buf);

	readPNM(file, buf);
	if(atoi(buf) > UCHAR_MAX)
	{
		cerr << "Unable to open '" << name << "'." << endl;
		throw errorPNM();
	}

	// Read data
	image<unsigned char>* im = new image<unsigned char>(width, height);
	file.read((char*)imPtr(im, 0, 0), width*height*sizeof(unsigned char));

	return im;
}


void readPNM(ifstream &file, char* buf)
{
	char doc[BUF_SIZE];
	char c;

	file >> c;
	while (c == '#')
	{
		file.getline(doc, BUF_SIZE);
		file >> c;
	}
	file.putback(c);

	file.width(BUF_SIZE);
	file >> buf;
	file.ignore();
}


void savePPM(image<Color>* im, const char* name)
{
	int width = im->width();
	int height = im->height();
	ofstream file(name, ios::out | ios::binary);

	file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
	file.write((char*)imPtr(im, 0, 0), width*height*sizeof(Color));
}


Color randomColor()
{
	Color c;
	c.r = (unsigned char) rand();
	c.g = (unsigned char) rand();
	c.b = (unsigned char) rand();

	return c;
}


__global__ void evolveContour(unsigned char* intensity, unsigned char* labels, signed char* speed, signed char* phi, int HEIGHT, int WIDTH, int* targetLabels, int kernelID, int numLabels, int* lowerIntensityBounds, int* upperIntensityBounds)
{
        int tid = threadIdx.x;

        intensity = &intensity[kernelID*HEIGHT*WIDTH];
        labels = &labels[kernelID*HEIGHT*WIDTH];
        speed = &speed[(kernelID*numLabels+tid)*HEIGHT*WIDTH];
        phi = &phi[(kernelID*numLabels+tid)*HEIGHT*WIDTH];

        dim3 dimGrid(WIDTH/30+1, HEIGHT/30+1);
        dim3 dimBlock(32, 32);
        initSpeedPhi<<<dimGrid, dimBlock>>>(intensity, labels, speed, phi, HEIGHT, WIDTH, targetLabels[tid], lowerIntensityBounds[tid], upperIntensityBounds[tid]);

        int numIterations = 0;
        stopCondition[tid] = 1;
        while(stopCondition[tid])
        {
                stopCondition[tid] = 0;
                numIterations++;

                dimGrid.x = WIDTH/30+1;
                dimGrid.y = HEIGHT/30+1;
 
		// Outward evolution
                switchIn<<<dimGrid, dimBlock>>>(speed, phi, HEIGHT, WIDTH);

                // Inward evolution
                switchOut<<<dimGrid, dimBlock>>>(speed, phi, HEIGHT, WIDTH);

                // Check stopping condition on every third iteration
                if(numIterations % 3 == 0)
                {
                        dimGrid.x = WIDTH/32+1;
                        dimGrid.y = HEIGHT/32+1;
                        checkStopCondition<<<dimGrid, dimBlock>>>(speed, phi, tid, HEIGHT, WIDTH);
                        cudaDeviceSynchronize();
                }
		else
			stopCondition[tid] = 1;

                if(stopCondition[tid] == 0)
                	printf("Target label %d (intensities: %d-%d) converged in %d iterations.\n", targetLabels[tid], lowerIntensityBounds[tid], upperIntensityBounds[tid], numIterations);
	}
}


__global__ void initSpeedPhi(unsigned char* intensity, unsigned char* labels, signed char* speed, signed char* phi, int HEIGHT, int WIDTH, int targetLabel, int lowerIntensityBound, int upperIntensityBound)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 30*bx + tx;
	int yPos = 30*by + ty;

	int intensityReg;
	int speedReg;
	int phiReg;
	__shared__ int labelsTile[32][32];

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		labelsTile[ty][tx] = labels[yPos*WIDTH+xPos];
		intensityReg = intensity[yPos*WIDTH+xPos];
	}

	// Initialization
	if(tx > 0 && tx < 31 && ty > 0 && ty < 31 && xPos < WIDTH-1 && yPos < HEIGHT-1)
	{
		// Phi
		if(labelsTile[ty][tx] != targetLabel)
		{
			if(labelsTile[ty][tx-1] != targetLabel && labelsTile[ty][tx+1] != targetLabel && labelsTile[ty-1][tx] != targetLabel && labelsTile[ty+1][tx] != targetLabel)
				phiReg = 3;
			else
				phiReg = 1;
		}
		else
		{
			if(labelsTile[ty][tx-1] != targetLabel || labelsTile[ty][tx+1] != targetLabel || labelsTile[ty-1][tx] != targetLabel || labelsTile[ty+1][tx] != targetLabel)
				phiReg = -1;
			else
				phiReg = -3;
		}

		// Speed
		if(intensityReg >= lowerIntensityBound && intensityReg <= upperIntensityBound)
			speedReg = 1;
		else
			speedReg = -1;

		// Load data back into global memory
		speed[yPos*WIDTH+xPos] = speedReg;
		phi[yPos*WIDTH+xPos] = phiReg;
	}
}


__global__ void switchIn(signed char* speed, signed char* phi, int HEIGHT, int WIDTH)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 30*bx + tx;
	int yPos = 30*by + ty;

	int speedReg;
	__shared__ int phiTile[32][32];

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		speedReg = speed[yPos*WIDTH+xPos];
		phiTile[ty][tx] = phi[yPos*WIDTH+xPos];
	}

	if(xPos > 0 && xPos < WIDTH-1 && yPos > 0 && yPos < HEIGHT-1)
	{
		// Delete points from Lout and add them to Lin
		if(phiTile[ty][tx] == 1 && speedReg > 0)
			phiTile[ty][tx] = -1;

		if(tx > 0 && tx < 31 && ty > 0 && ty < 31)
		{
			// Update neighborhood
			if(phiTile[ty][tx] == 3)
			{
				if(phiTile[ty][tx-1] == -1 || phiTile[ty][tx+1] == -1 || phiTile[ty-1][tx] == -1 || phiTile[ty+1][tx] == -1)
					phiTile[ty][tx] = 1;
			}

			// Eliminate redundant points in Lin
			if(phiTile[ty][tx] == -1)
			{
				if(phiTile[ty][tx-1] < 0 && phiTile[ty][tx+1] < 0 && phiTile[ty-1][tx] < 0 && phiTile[ty+1][tx] < 0)
					phiTile[ty][tx] = -3;
			}

			// Load data back into global memory
			phi[yPos*WIDTH+xPos] = phiTile[ty][tx];
		}
	}
}


__global__ void switchOut(signed char* speed, signed char* phi, int HEIGHT, int WIDTH)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 30*bx + tx;
	int yPos = 30*by + ty;

	int speedReg;
	__shared__ int phiTile[32][32];

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		speedReg = speed[yPos*WIDTH+xPos];
		phiTile[ty][tx] = phi[yPos*WIDTH+xPos];
	}

	if(xPos > 0 && xPos < WIDTH-1 && yPos > 0 && yPos < HEIGHT-1)
	{
		// Delete points from Lin and add them to Lout
		if(phiTile[ty][tx] == -1 && speedReg < 0)
			phiTile[ty][tx] = 1;

		if(tx > 0 && tx < 31 && ty > 0 && ty < 31)
		{
			// Update neighborhood
			if(phiTile[ty][tx] == -3)
			{
				if(phiTile[ty][tx-1] == 1 || phiTile[ty][tx+1] == 1 || phiTile[ty-1][tx] == 1 || phiTile[ty+1][tx] == 1)
					phiTile[ty][tx] = -1;
			}

			// Eliminate redundant points
			if(phiTile[ty][tx] == 1)
			{
				if(phiTile[ty][tx-1] > 0 && phiTile[ty][tx+1] > 0 && phiTile[ty-1][tx] > 0 && phiTile[ty+1][tx] > 0)
					phiTile[ty][tx] = 3;
			}

			// Load data back into global memory
			phi[yPos*WIDTH+xPos] = phiTile[ty][tx];
		}
	}

}


__global__ void checkStopCondition(signed char* speed, signed char* phi, int parentThreadID, int HEIGHT, int WIDTH)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int xPos = 32*bx + tx;
	int yPos = 32*by + ty;

	int speedReg;
	int phiReg;

	// Load data into shared memory and registers
	if(xPos < WIDTH && yPos < HEIGHT)
	{
		speedReg = speed[yPos*WIDTH+xPos];
		phiReg = phi[yPos*WIDTH+xPos];
	}

	// Falsify stop condition if criteria are not met
	if(phiReg == 1 && speedReg > 0)
		stopCondition[parentThreadID]++;
	else if(phiReg == -1 && speedReg < 0)
		stopCondition[parentThreadID]++;
}
