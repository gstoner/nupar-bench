/*

LoKDR algorithm implementation on a GPU using the KNN implementation provided by Garcia et al.

*/

/**
  *
  * Date         April 5, 2012
  * ====
  *
  * Authors      Ayse Yilmazer
  * =======      Fatemeh Azmandian
  *              
  *
  * Description  Given a dataset, performs Local Kernel Density Ratio (LoKDR) feature selection  
  * ===========  and returns a list of features sorted in descending order of their importance
  *              for outlier detection. The NVIDIA CUDA API is used to speed up the algorthim.
  *              
  *
  *
  *		ORGANIZATION OF DATA
  *		====================
  *
  *		In CUDA, it is usual to use the notion of array.
  *		For our program, the following array
  *	
  *			A = | 1 3 5 |
  *				| 2 4 6 |
  *	
  *		corresponds to the a set of 3 points of dimension 2:
  *	
  *			p1 = (1, 2)
  *			p2 = (3, 4)
  *			p3 = (5, 6)
  *		
  *		The array A is actually stored in memory as a linear vector:
  *	
  *			A = (1, 3, 5, 2, 4, 6)
  * 
  *
  */

// Includes
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <limits.h>

#include "cuda.h"
#include "cuPrintf.cu"
//#include "cublas.h"	/* COMMENTED OUT BY FATEMEH since we are not currently using CUBLAS */

#include "float.h"		// For the DBL_MIN and DBL_MAX		/* ADDED BY FATEMEH */  
#include "args.h"		// For easier argument parsing
#include "err.h"
#include "parsing.h"

#include "lokdr.h"

//#define PRINT_PROGRESS 
#define LESS_COMPS_SOME_COPYS 

#ifndef MAX
#define MAX(x,y) (((x)>(y))?(x):(y))
#endif 

#ifndef ABS
#define ABS(x) (((x)>0)?(x):(-(x)))
#endif 

#ifdef _WIN32
#pragma warning(disable:4996)
#pragma warning(disable:4101)
#define PATH_SEPARATOR "\\"
#else 
#define PATH_SEPARATOR "/"
#endif

// Constants used by the program
#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9
#define BLOCK_DIM                      16

// Texture containing the reference points (if it is possible)	/* ADDED BY FATEMEH (from the CUDA version) */
texture<float, 2, cudaReadModeElementType> texA;


#define MAX_FILE_NAME_SIZE 1024	
#define MAX_FEATURE_COUNT 15000

// Some structures used
//typedef void*		userData;

/* ADDED BY FATEMEH */
// A function to calculate the elapsed time
char *get_elapsed(float sec);

/* ADDED BY FATEMEH (from the CUDA version) */
//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//



/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  * The matrix A is a texture.
  *
  * @param first_round	specifies if this is the first round of the feature selection, in which case all the pairwise distances need to be calculated; otherwise, the distances from the last round will be used and the contribution of a feature will be added/removed
  * @param wA			width of the matrix A = number of points in A
  * @param B			pointer on the matrix B
  * @param wB			width of the matrix B = number of points in B
  * @param pB			pitch of matrix B given in number of columns
  * @param dim			dimension of points = height of matrices A and B
  * @param AB			pointer on the matrix containing the wA*wB distances computed
  * @param pAB			pitch of matrix AB given in number of columns						// ADDED BY FATEMEH
  * @param ufm_dev		pointer to the matrix containing the set of features to use			// ADDED BY FATEMEH
  * @param ufm_pitch	the number of columns in the use_feature_matrix on the device		// ADDED BY FATEMEH
  * @param feature_combinations  the number of features combinations (or sets)				// ADDED BY FATEMEH
  */
__global__ void cuComputeDistanceTexture(int first_round, int wA, float * B, int wB, int pB, int dim, float* AB, int pAB, int *ufm_dev, int ufm_pitch, int feature_combos){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int feature_set = xIndex/wB;				/* ADDED BY FATEMEH */
	
	if ( (xIndex < feature_combos * wB) && (yIndex < wA) ){	/* MODIFIED BY FATEMEH (the "feature_combos *") */
	    float ssd = 0;
#ifdef LESS_COMPS_SOME_COPYS
		if (first_round || feature_combos == 1)	// If (feature_combos == 1), it means that we have just removed a bunch of features and we want to calculate the criterion function for the new (one and only) set of features, so we can't use the incremental distances from a previous round of the feature search and we need to calculate them from scratch
		{
#endif
			for (int i=0; i<dim; i++){
				if (ufm_dev[feature_set*ufm_pitch + i])	// If the value representing the i-th feature in the use_feature_matrix on the device is not zero, then apply the contribution of the feature to calculate the distance /* ADDED BY FATEMEH */
				{
					float tmp  = tex2D(texA, (float)yIndex, (float)i) - B[i * pB + (xIndex % wB)];	/* MODIFIED BY FATEMEH (the % wB) */
					ssd += tmp * tmp;
				}
			}
			AB[yIndex * pAB + xIndex] = ssd;		/* MODIFIED BY FATEMEH (the pAB) */
			//DEBUGGING!!!!!!!!!!!!
			//printf("xIndex: %d \t AB[yIndex * pAB + xIndex]: %f\n", xIndex, AB[yIndex * pAB + xIndex]);
#ifdef LESS_COMPS_SOME_COPYS
		}
		else
		{
			int multiplier = ufm_dev[feature_set*ufm_pitch];			// Whether we should add the feature (+1) or remove it (-1)
			int changed_feature = ufm_dev[feature_set*ufm_pitch + 1];	// The index of the feature to be added/removed
			
			float tmp  = tex2D(texA, (float)yIndex, (float)changed_feature) - B[changed_feature * pB + (xIndex % wB)];	/* MODIFIED BY FATEMEH (the % wB) */
			AB[yIndex * pAB + xIndex] += multiplier * tmp * tmp;

			//DEBUGGING!!!!!!!!!!!!
			//printf("xIndex: %d \t AB[yIndex * pAB + xIndex]: %f\n", xIndex, AB[yIndex * pAB + xIndex]);
		}
#endif
    }
}


/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param first_round	specifies if this is the first round of the feature selection, in which case all the pairwise distances need to be calculated; otherwise, the distances from the last round will be used and the contribution of a feature will be added/removed
  * @param A			pointer on the matrix A
  * @param wA			width of the matrix A = number of points in A
  * @param pA		    pitch of matrix A given in number of columns
  * @param B			pointer on the matrix B
  * @param wB			width of the matrix B = number of points in B
  * @param pB			pitch of matrix B given in number of columns
  * @param dim			dimension of points = height of matrices A and B
  * @param AB			pointer on the matrix containing the wA*wB distances computed
  * @param pAB			pitch of matrix AB given in number of columns					// ADDED BY FATEMEH
  * @param ufm_dev		pointer to the matrix containing the set of features to use		// ADDED BY FATEMEH
  * @param ufm_pitch	the number of columns in the use_feature_matrix on the device	// ADDED BY FATEMEH
  * @param feature_combinations  the number of features combinations (or sets)			// ADDED BY FATEMEH
  */
__global__ void cuComputeDistanceGlobal(int first_round, float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB, int pAB, int *ufm_dev, int ufm_pitch, int feature_combos){

	// Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
	__shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
	__shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];
    
    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B_AB;
	__shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;
	
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	
	// Other variables
	float tmp;
    float ssd = 0;

#ifdef LESS_COMPS_SOME_COPYS
	if (first_round || feature_combos == 1)	// If (feature_combos == 1), it means that we have just removed a bunch of features and we want to calculate the criterion function for the new (one and only) set of features, so we can't use the incremental distances from a previous round of the feature search and we need to calculate them from scratch
	{
#endif
		// Loop parameters
		begin_A = BLOCK_DIM * blockIdx.y;
		begin_B_AB = BLOCK_DIM * blockIdx.x;	/* MODIFIED BY FATEMEH */	// The x-dimension beginning of the AB matrix computations
		begin_B = begin_B_AB % wB;				/* MODIFIED BY FATEMEH */
		step_A  = BLOCK_DIM * pA;
		step_B  = BLOCK_DIM * pB;
		end_A   = begin_A + (dim-1) * pA;

		// Conditions
		int cond0 = (begin_A + tx < wA); // used to write in shared memory
		int cond1 = (begin_B_AB + tx < wB * feature_combos); //(begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
		int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix
		//int cond3 = (begin_B_AB + tx < wB * feature_combos); // used to write in shared memory & to computations and to write in output matrix
	    
		// Loop over all the sub-matrices of A and B required to compute the block sub-matrix
		for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
	        
			// Load the matrices from device memory to shared memory; each thread loads one element of each matrix
			if (a/pA + ty < dim){
				shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
				shared_B[ty][tx] = (cond1)? B[((begin_B + tx) % wB) + pB * ty + (b-begin_B)] : 0;
				// DEBUGGING!!!!!!!!!!!!!!!!!!
				//if (blockIdx.x == 2446 && blockIdx.y == 0 && threadIdx.x == 14 && threadIdx.y == 7)
				//	cond1 = (begin_B_AB + tx < wB * feature_combos);
			}
			else{
				shared_A[ty][tx] = 0;
				shared_B[ty][tx] = 0;
			}
	        
			// Synchronize to make sure the matrices are loaded
			__syncthreads();
	        
			// Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
			unsigned int feature_set = (begin_B_AB + tx)/wB;		/* ADDED BY FATEMEH */
			if (cond2 && cond1){
				for (int k = 0; k < BLOCK_DIM; ++k){
					if (ufm_dev[feature_set*ufm_pitch + a/pA + k])	// If the value representing the (a/pA + k)-th feature in the use_feature_matrix on the device is not zero, then apply the contribution of the feature to calculate the distance /* ADDED BY FATEMEH */
					{
						tmp = shared_A[k][ty] - shared_B[k][tx];//[tx];	((begin_B_AB + tx) % wB) % BLOCK_DIM
						ssd += tmp*tmp;
						// DEBUGGING!!!!!!!!!!!!!!!!!!
						//if (blockIdx.x == 2446 && blockIdx.y == 0 && threadIdx.x == 14 && threadIdx.y == 0)
						//	cond1 = (begin_B_AB + tx < wB * feature_combos);//(begin_B + tx < wB);

					}
				}
			}
	        
			// Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
			__syncthreads();
		}
	    
		// DEBUGGING!!!!!!!!!!!!!!!!!!
		//if (blockIdx.x == 2446 && blockIdx.y == 0 && threadIdx.x == 14 && threadIdx.y == 0)
		//	cond1 = (begin_B_AB + tx < wB * feature_combos);

		// Write the block sub-matrix to device memory; each thread writes one element
		if (cond2 && cond1)
			AB[ (begin_A + ty) * pAB + begin_B_AB + tx ] = ssd;	/* MODIFIED BY FATEMEH (the pAB) */
#ifdef LESS_COMPS_SOME_COPYS
	}
	else
	{
		unsigned int xIndex = blockIdx.x * blockDim.x + tx;	// cond2
		unsigned int yIndex = blockIdx.y * blockDim.y + ty;	// cond1
		if ((xIndex < feature_combos * wB) && (yIndex < wA))
		{
			unsigned int feature_set = xIndex/wB;				/* ADDED BY FATEMEH */
			
			int multiplier = ufm_dev[feature_set*ufm_pitch];			// Whether we should add the feature (+1) or remove it (-1)
			int changed_feature = ufm_dev[feature_set*ufm_pitch + 1];	// The index of the feature to be added/removed
			
			float tmp  = A[changed_feature * pA + yIndex] - B[changed_feature * pB + (xIndex % wB)];	/* MODIFIED BY FATEMEH (the % wB) */
			AB[yIndex * pAB + xIndex] += multiplier * tmp * tmp;
		}
	}
#endif
}

// ADDED BY FATEMEH ////////////////////////////////////////////////////////////////
/**
  * Copies matrix_src into matrix_dest
  *
  * @param matrix_dest		destination matrix
  * @param matrix_src		source matrix
  * @param p_dest			pitch of the destination matrix
  * @param p_src			pitch of the source matrix
  * @param width			width to copy
  * @param height			height to copy
  */
__global__ void cuCopyMatrix(float *matrix_dest, float *matrix_src, int p_dest, int p_src, int width, int height){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (xIndex < width) && (yIndex < height) )
		matrix_dest[yIndex * p_dest + xIndex] = matrix_src[yIndex * p_src + xIndex];
	//DEBUGGING!!!!!!!!!!!!
	//printf("xIndex: %d \t matrix_dest[yIndex * p_dest + xIndex]: %f\n", xIndex, matrix_dest[yIndex * p_dest + xIndex]);
}

////////////////////////////////////////////////////////////////////////////////////

// ADDED BY FATEMEH ////////////////////////////////////////////////////////////////
/**
  * Duplicates first "width" columns of matrix into remaining part, "duplicate" times
  *
  * @param matrix			pointer to the matrix
  * @param p_matrix			pitch of the matrix
  * @param width			width to copy
  * @param height			height to copy
  * @param duplicate		number of times to duplicate
  */
__global__ void cuDuplicateMatrix(float *matrix, int p_matrix, int width, int height, int duplicate){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (xIndex >= width) && (xIndex < duplicate*width) && (yIndex < height) )
		matrix[yIndex * p_matrix + xIndex] = matrix[yIndex * p_matrix + (xIndex % width)];
	//DEBUGGING!!!!!!!!!!!!
	//printf("xIndex: %d \t matrix_dest[yIndex * p_dest + xIndex]: %f\n", xIndex, matrix_dest[yIndex * p_dest + xIndex]);
}

////////////////////////////////////////////////////////////////////////////////////


/**
  * Gathers k-th smallest distances for each column of the distance matrix at the top.
  *
  * @param dist     distance matrix
  * @param width    width of the distance matrix
  * @param pitch    pitch of the distance matrix given in number of columns
  * @param height   height of the distance matrix
  * @param k        number of smallest distance to consider
  */
__global__ void cuInsertionSort(float *dist, int width, int pitch, int height, int k){

	// Variables
    int l,i,j;
    float *p;
    float v, max_value;
    //int local_pitch; //AYSE: defined 

	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;	
    //local_pitch = pitch*feature_combos;			/* ADDED BY FATEMEH */ // This is done because the way I envision the distance matrices for different feature sets is having them side by side: |dist_mat_1||dist_mat_2|...|dist_mat_feature_combos|
    
	if (xIndex < width){
        
        // Pointer shift and max value
        p         = dist+xIndex;
        max_value = *p;
        
        // Part 1 : sort kth first elements
        for (l=pitch;l<k*pitch;l+=pitch){
            v = *(p+l);
            if (v<max_value){
                i=0; while (i<l && *(p+i)<=v) i+=pitch;
                for (j=l;j>i;j-=pitch)
                    *(p+j) = *(p+j-pitch);
                *(p+i) = v;
            }
            max_value = *(p+l);
        }
        
        // Part 2 : insert element in the k-th first lines
        for (l=k*pitch;l<height*pitch;l+=pitch){
            v = *(p+l);
            if (v<max_value){
                i=0; while (i<k*pitch && *(p+i)<=v) i+=pitch;
                for (j=(k-1)*pitch;j>i;j-=pitch)
                    *(p+j) = *(p+j-pitch);
                *(p+i) = v;
                max_value  = *(p+(k-1)*pitch);
            }
        }
    }
}

// ADDED BY FATEMEH: The following is from the CUDA with indices version, which calculates the square root of the first k rows
/**
  * Computes the square root of the first line (width-th first element)
  * of the distance matrix.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param pitch   pitch of the distance matrix given in number of columns
  * @param k       number of neighbors to consider
  * @param x_threads_count   the number of threads in the x dimension before considering the different number of features (used to calculate which feature set to use)	// ADDED BY FATEMEH
  */
__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k, int x_threads_count){

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    
	if (xIndex % x_threads_count < width && yIndex<k)			/* MODIFIED BY FATEMEH (the % x_threads_count) */
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}


/* ADDED BY FATEMEH */
/**
 * Given a matrix of the k-nearest neighbor distances, 
 * calculates the kernel density estimate of each datapoint. 
 * The result is stored in the first row of the matrix.
 *
 * @param mat    : the matrix of k-nearest neighbor distances
 * @param width  : the number of columns
 * @param pitch  : the pitch in number of columns
 * @param k      : the number of neighbors to consider
 * @param sigma  : the width of the Gaussian kernel
 *  
 */
__global__ void parallelKDE(float *mat, int width, int pitch, int k, float sigma){

	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
	//DEBUGGING!!!!!!!!!!!!
	//cuPrintf("xIndex: %d \t mat[xIndex]: %f\n", xIndex, mat[xIndex]);
    
	if (xIndex < width)
	{
		float sum = 0;
		for (int i = 1; i < k; i++)		// We intentionally start from i=1 because the point is always going to be its own nearest neighbor (the way we use it)
		{
			sum += exp(-mat[i*pitch + xIndex]/(2*sigma*sigma));
			//cuPrintf("xIndex: %d \t i %d \t pitch %d \t i*pitch + xIndex %d \t mat[i*pitch + xIndex]: %f\n", xIndex, i, pitch, i*pitch + xIndex, mat[i*pitch + xIndex]);
		}
		mat[xIndex] = sum;

		//DEBUGGING!!!!!!!!!!!!
		//cuPrintf("xIndex: %d \t kde: %f\n", xIndex, sum);
    }
}



//-----------------------------------------------------------------------------------------------//
//                                   Sort Criterion Values	                                     //
//-----------------------------------------------------------------------------------------------//


// FATEMEH's COMMENT: This can be parallelized as well but might not be worth the cost of moving to/from global memory and accessing global memory 
/**
  * Sorts the criterion values for different feature sets (using insertion sort)
  *
  * @param criterions       criterion values for different feature combinations (sets)
  * @param indices			corresponding indices of the features
  * @param elements			the number of elements to sort
  */
void sort_criterions(float *criterions, int *indices, int elements)		// Sorts in descending order
{
	int i, j, index;
	float value;
	
	for (i = 1; i < elements; i++)
	{
		value = criterions[i];
		index = indices[i];
		j = i - 1;
		while (j >= 0 && criterions[j] < value)
		{
			criterions[j+1] = criterions[j];
			indices[j+1] = indices[j];
			j--;
		}
		criterions[j+1] = value;
		indices[j+1] = index;
	}
}


//-----------------------------------------------------------------------------------------------//
//                                   Calculate LoKDR Criterion                                   //
//-----------------------------------------------------------------------------------------------//



/**
  * Prints the error message return during the memory allocation.
  *
  * @param error        error value return by the memory allocation function
  * @param memorySize   size of memory tried to be allocated
  */
void printErrorMessage(cudaError_t error, int memorySize){
    printf("==================================================\n");
    printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
    printf("Requested memory to be allocated: %d\n", memorySize);
    printf("==================================================\n");
}



/**
  * Calculate criterion for the LoKDR feature selection
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distance to the k-th nearest neighbor for each query point
  * - Copy distances from device to host memory
  *
  * @param first_round   specifies if this is the first round of the feature selection, in which case all the pairwise distances need to be calculated; otherwise, the distances from the last round will be used and the contribution of a feature will be added/removed
  * @param criterion_val the criterion value to which we need to compare the best criterion we find here in order to determine whether to add/remove a feature      
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k-th nearest neighbor ; pointer to linear matrix
  *
  */
void calculate_LoKDR_criterion(int first_round, float criterion_val, float *ref_host, int ref_width, float *query_host, int query_width, /* ADDED BY FATEMEH */ int *use_features_matrix, /* ADDED BY FATEMEH */ int feature_combinations, int height, int k, /* ADDED BY FATEMEH */ float sigma, gpu_args *criterion_args)	
{
    unsigned int size_of_float = sizeof(float);    
	int i, j;						/* ADDED BY FATEMEH */
	int first_changed_feature;		// First feature that will be either added or removed in the feature space search
	int multiplier;					// A value that will be multiplied by negative values to make them positive

	//cudaEvent_t function_start, function_end;
	//cudaEventCreate(&function_start);
	//cudaEventCreate(&function_end);
	//float elapsed_time;

	for (i = 0; i < feature_combinations; i++)
	{
		criterion_args->kde_sum_normal[i] = 0;		
		criterion_args->kde_sum_abnormal[i] = 0;	
	}

	// Memory copy of use_feature_matrix in ufm_dev
	// Note that if this is the first round of the feature selection, we copy all columns of use_features_matrix (with each column corresponding to a feature); otherwise, we only copy the first two columns of use_features_matrix (the first column tells us whether the feature should be added (+1) or removed (-1) and in the second column, we have the index of the feature to change)
#ifdef LESS_COMPS_SOME_COPYS
	cudaMemcpy2D(criterion_args->ufm_dev, criterion_args->ufm_pitch_in_bytes, use_features_matrix, height*size_of_float, (first_round || (feature_combinations == 1)? height : 2)*size_of_float, feature_combinations, cudaMemcpyHostToDevice);
#else
	cudaMemcpy2D(criterion_args->ufm_dev, criterion_args->ufm_pitch_in_bytes, use_features_matrix, height*size_of_float, height*size_of_float, feature_combinations, cudaMemcpyHostToDevice);
#endif
	/************************************************ ADDED BY FATEMEH */

    // Main loop: split queries to fit in GPU memory
    for (i=0;i<query_width;i+=criterion_args->max_nb_query_treated){
        
        // Number of query points actually used
        criterion_args->actual_nb_query_width = min((unsigned int)criterion_args->max_nb_query_treated, query_width-i);	/* MODIFIED BY FATEMEH (casting to "unsigned int") */
        
        // Copy part of query actually being treated
        cudaMemcpy2D(criterion_args->query_dev, criterion_args->query_pitch_in_bytes, &query_host[i], query_width*size_of_float, criterion_args->actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice); // NOTE: I decided to comment out "+1" because we don't use the label on the GPU
	
		// Grids and threads
		dim3 g_BLOCK_DIMxBLOCK_DIM(feature_combinations*criterion_args->actual_nb_query_width/BLOCK_DIM, ref_width/BLOCK_DIM, 1);	/* MODIFIED BY FATEMEH */
        dim3 t_BLOCK_DIMxBLOCK_DIM(BLOCK_DIM, BLOCK_DIM, 1);
        if (feature_combinations*criterion_args->actual_nb_query_width % BLOCK_DIM != 0) g_BLOCK_DIMxBLOCK_DIM.x += 1;		/* MODIFIED BY FATEMEH */
        if (ref_width % BLOCK_DIM != 0) g_BLOCK_DIMxBLOCK_DIM.y += 1;
        //
		dim3 g_duplicate(criterion_args->actual_nb_query_width/BLOCK_DIM, ref_width/BLOCK_DIM, 1);	/* MODIFIED BY FATEMEH */
        if (criterion_args->actual_nb_query_width % BLOCK_DIM != 0) g_duplicate.x += 1;		/* MODIFIED BY FATEMEH */
        if (ref_width % BLOCK_DIM != 0) g_duplicate.y += 1;
        //
		dim3 g_BLOCK_DIM2x1(feature_combinations*criterion_args->actual_nb_query_width/(BLOCK_DIM*BLOCK_DIM), 1, 1);			/* MODIFIED BY FATEMEH (the feature_combinations) */ // ASK AYSE: Do we have to check to see that the number of blocks does not exceed the maximum allowed? AYSE: Fatemeh, We should check if it could exceed the max number of threads allowed
        dim3 t_BLOCK_DIM2x1((BLOCK_DIM*BLOCK_DIM), 1, 1);
        if (feature_combinations*criterion_args->actual_nb_query_width % (BLOCK_DIM*BLOCK_DIM) != 0) g_BLOCK_DIM2x1.x += 1;	/* MODIFIED BY FATEMEH (the feature_combinations) */ 
		
#ifdef LESS_COMPS_SOME_COPYS
		if (!first_round && feature_combinations > 1)	// When it is the first round, or feature_combinations == 1, we will not be reusing distance computations from the previous round, we will be calculating all the distances
		{
			//cudaEventRecord(function_start, 0);		// Start the timer for the beginning of the feature selection

			// Copy dist_host, which contains distances for the current feature combination, to dist_dev
			cudaMemcpy2D(criterion_args->dist_dev, criterion_args->dist_pitch_in_bytes, &(criterion_args->dist_host[i]), query_width*size_of_float, criterion_args->actual_nb_query_width*size_of_float, ref_width, cudaMemcpyHostToDevice);
   
			/*cudaEventRecord(function_end, 0);
			cudaEventSynchronize(function_end);
			cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
			//printf("Elapsed time for copying from the host to dist_dev: %s\n", get_elapsed(elapsed_time/1000));

			//cudaEventRecord(function_start, 0);		// Start the timer for the beginning of the feature selection

			// NOTE: I decided not to use the following because it was slower than the function cuDuplicateMatrix
			//for (j = 1; j < feature_combinations; j++)
			//{
			//	// Copy the distances in the part of dist_dev that corresponds to the first feature combination into sections corresponding to subsequent feature combinations
			//	cuCopyMatrix<<<g_duplicate,t_BLOCK_DIMxBLOCK_DIM>>>(&(criterion_args->dist_dev[j*criterion_args->actual_nb_query_width]), &(criterion_args->dist_dev[0]), criterion_args->dist_pitch, criterion_args->dist_pitch, criterion_args->actual_nb_query_width, ref_width);
			//	
			//	////DEBUGGING!!!!!!!!!!!!!!
			//	//cudaThreadSynchronize();
			//}

			cuDuplicateMatrix<<<g_BLOCK_DIMxBLOCK_DIM,t_BLOCK_DIMxBLOCK_DIM>>>(criterion_args->dist_dev, criterion_args->dist_pitch, criterion_args->actual_nb_query_width, ref_width, feature_combinations);

			/*cudaEventRecord(function_end, 0);
			cudaEventSynchronize(function_end);
			cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
			//printf("Elapsed time for duplicating dist_dev: %s\n", get_elapsed(elapsed_time/1000));
		}
#endif

		// Kernel 1: Compute all the distances
		//cudaEventRecord(function_start, 0);		// Start the timer for the beginning of the feature selection

		if (criterion_args->use_texture)
			cuComputeDistanceTexture<<<g_BLOCK_DIMxBLOCK_DIM,t_BLOCK_DIMxBLOCK_DIM>>>(first_round, ref_width, criterion_args->query_dev, criterion_args->actual_nb_query_width, criterion_args->query_pitch, height, criterion_args->dist_dev, criterion_args->dist_pitch, criterion_args->ufm_dev, criterion_args->ufm_pitch, feature_combinations);	/* MODIFIED BY FATEMEH */
		else
			cuComputeDistanceGlobal<<<g_BLOCK_DIMxBLOCK_DIM,t_BLOCK_DIMxBLOCK_DIM>>>(first_round, criterion_args->ref_dev, ref_width, criterion_args->ref_pitch, criterion_args->query_dev, criterion_args->actual_nb_query_width, criterion_args->query_pitch, height, criterion_args->dist_dev, criterion_args->dist_pitch, criterion_args->ufm_dev, criterion_args->ufm_pitch, feature_combinations);	/* MODIFIED BY FATEMEH */

		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for cuComputeDistance kernel: %s\n", get_elapsed(elapsed_time/1000));

		////DEBUGGING!!!!!!!!!!!!!!
		//cudaPrintfDisplay(stdout, true);

		////DEBUGGING!!!!!!!!!!!!!!
		//float *sdist_host = (float *) malloc(criterion_args->actual_nb_query_width * feature_combinations * ref_width * sizeof(float));	/* ADDED BY FATEMEH */

		//cudaMemcpy2D(sdist_host, criterion_args->actual_nb_query_width * feature_combinations* sizeof(float), criterion_args->dist_dev, criterion_args->dist_pitch_in_bytes, criterion_args->actual_nb_query_width * feature_combinations * sizeof(float), ref_width, cudaMemcpyDeviceToHost);
	
		//for (j = 0; j < ref_width; j++)
		//	printf("%f ", sdist_host[j*criterion_args->actual_nb_query_width * feature_combinations + 87*criterion_args->actual_nb_query_width]);
		//free(sdist_host);

		//////////////////////////
#ifdef LESS_COMPS_SOME_COPYS
		//cudaEventRecord(function_start, 0);		// Start the timer 

		// Copy the part of dist_dev representing the first feature combination back to dist_host (will be re-used in subsequent rounds of the feature search)
		cudaMemcpy2D(&(criterion_args->dist_host[i]), query_width*size_of_float, criterion_args->dist_dev, criterion_args->dist_pitch_in_bytes, criterion_args->actual_nb_query_width*size_of_float, ref_width, cudaMemcpyDeviceToHost);   

		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for copying part of dist_dev to the host: %s\n", get_elapsed(elapsed_time/1000));
#endif
		//cudaEventRecord(function_start, 0);		// Start the timer 

        // Kernel 2: Sort each column (the k-nearest neighbor distances)
        cuInsertionSort<<<g_BLOCK_DIM2x1,t_BLOCK_DIM2x1>>>(criterion_args->dist_dev, feature_combinations * criterion_args->actual_nb_query_width, criterion_args->dist_pitch, ref_width, k+1);	/* MODIFIED BY FATEMEH */	// The "k+1" is because a point is always its own nearest neighbor
        
		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for cuInsertionSort kernel: %s\n", get_elapsed(elapsed_time/1000));

		////DEBUGGING!!!!!!!!!!!!!!
		//float *sdist_host = (float *) malloc(criterion_args->actual_nb_query_width * feature_combinations * ref_width * sizeof(float));	/* ADDED BY FATEMEH */

		//cudaMemcpy2D(sdist_host, criterion_args->actual_nb_query_width * feature_combinations* sizeof(float), criterion_args->dist_dev, criterion_args->dist_pitch_in_bytes, criterion_args->actual_nb_query_width * feature_combinations * sizeof(float), ref_width, cudaMemcpyDeviceToHost);
	
		//for (j = 0; j < ref_width; j++)
		//	printf("%f ", sdist_host[j*criterion_args->actual_nb_query_width * feature_combinations]);
		//free(sdist_host);

		//////////////////////////

		/* ADDED BY FATEMEH ************************************************/ 

		// Now we have the squared distances ready in dist_dev
		// Eeach thread calculates the density of one point, which requires summing over its k-nearest neighbors
		
		//cudaEventRecord(function_start, 0);		// Start the timer 

		parallelKDE<<<g_BLOCK_DIM2x1,t_BLOCK_DIM2x1>>>(criterion_args->dist_dev, feature_combinations * criterion_args->actual_nb_query_width, criterion_args->dist_pitch, k+1, sigma);	// The "k+1" is because a point is always its own nearest neighbor

		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for parallelKDE kernel: %s\n", get_elapsed(elapsed_time/1000));

		////DEBUGGING!!!!!!!!!!!!!!
		//cudaPrintfDisplay(stdout, true);

		// Now, the first row of the distance matrix (on the device) contains the KDE of all the query points

		//cudaEventRecord(function_start, 0);		// Start the timer 

		// Memory copy of output from device to host
		// The following will transfer the KDE of the query points
		for (j = 0; j < feature_combinations; j++)
			cudaMemcpy(&(criterion_args->kde_values[i+j*query_width]), &(criterion_args->dist_dev[j*criterion_args->actual_nb_query_width]), criterion_args->actual_nb_query_width*size_of_float, cudaMemcpyDeviceToHost);
	
		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for transfering the KDE values to the host: %s\n", get_elapsed(elapsed_time/1000));

		//DEBUGGING!!!!!!!
		/*printf("actual_nb_query_width: %d\n", actual_nb_query_width);
		printf("feature_combinations: %d\n", feature_combinations);
		for (int m = 0; m < feature_combinations; m++)
			for (j = 0; j < actual_nb_query_width; j++)
				printf("KDE in dist_host[%d][%d]: %f\n", m, i+j, dist_host[i+j+m*query_width]);
		*/

		/************************************************* ADDED BY FATEMEH */ 
	}

	/* ADDED BY FATEMEH ************************************************/ 

	//cudaEventRecord(function_start, 0);		// Start the timer 

	// In the C version of my code, I place the label after the last feature of the data matrix, which is what I will assume here too
	int label_row = query_width * height;
	for (i = 0; i < feature_combinations; i++)		// FATEMEH's COMMENT: This can be parallelized as well but might not be worth the cost of moving to/from global memory and accessing global memory 
	{
		for (j = 0; j < query_width; j++)
		{
			if (query_host[label_row + j] == -1)
				criterion_args->kde_sum_normal[i] += criterion_args->kde_values[i*query_width + j];
			else if (query_host[label_row + j] == 1)
				criterion_args->kde_sum_abnormal[i] += criterion_args->kde_values[i*query_width + j];
			else
			{
				printf("The label of the data point has not been correctly assigned.");
				exit(1);
			}
		}

		if (criterion_args->kde_sum_abnormal[i] != 0)
			criterion_args->feature_criterion_values[i] = criterion_args->kde_sum_normal[i]/criterion_args->kde_sum_abnormal[i];
		else criterion_args->feature_criterion_values[i] = DBL_MAX;

		//DEBUGGING!!!!!!!!!!!!!!!
		//printf("criterion_values[%d]: %f\n", i, criterion_values[i]);

		// Copy the part of dist_dev representing the first feature combination back to dist_host (will be re-used in subsequent steps of the feature search)
		//cudaMemcpy(&(criterion_args->dist_host[i]), &(criterion_args->dist_dev[0]), criterion_args->actual_nb_query_width*size_of_float, cudaMemcpyDeviceToHost);
	}  
	/*cudaEventRecord(function_end, 0);
	cudaEventSynchronize(function_end);
	cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
	//printf("Elapsed time for calculating the criterion values: %s\n", get_elapsed(elapsed_time/1000));

	if (feature_combinations > 1)	// The following does not need to be done when calculate_LoKDR_criterion is called just to calculate the criterion function for one feature set
	{
#ifdef LESS_COMPS_SOME_COPYS
		first_changed_feature = criterion_args->feature_indices[0];
		
		multiplier = use_features_matrix[0];	// The first column (and hence first element) of the use_feature_matrix specifies if we are adding or removing the feature
#endif

		//cudaEventRecord(function_start, 0);		// Start the timer 

		// Sort the criterion values
		sort_criterions(criterion_args->feature_criterion_values, criterion_args->feature_indices, feature_combinations);

		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for sorting the criterion values: %s\n", get_elapsed(elapsed_time/1000));

		// DEBUGGING!!!!!!!!!!!!
		/*for (i = 0; i < feature_combinations; i++)
			printf("feature_criterion_values[%d]: %f index: %d\n", i, criterion_args->feature_criterion_values[i], criterion_args->feature_indices[i]);
		printf("\n");*/

		// Now "the best feature to change" is in feature_indices[0]
		// If "the best feature to change" is different than "the first changed feature", modify dist_host to cancel the effect of "the first changed feature" and add the effect of "the best feature to change"

		//DEBUGGING!!!!!!!!!!!!!!!
		//printf("criterion_args->feature_indices[0]: %d \t first_changed_feature: %d\n", criterion_args->feature_indices[0], first_changed_feature); 

#ifdef LESS_COMPS_SOME_COPYS
		//cudaEventRecord(function_start, 0);		// Start the timer 

		if (criterion_val >= criterion_args->feature_criterion_values[0] || criterion_args->feature_indices[0] != first_changed_feature)
		{
			for (j = 0; j < ref_width; j++)
			{
				for (i = 0; i < query_width; i++)
				{
					float tmp1 = query_host[first_changed_feature*query_width + i] - ref_host[first_changed_feature*ref_width + j];
					float tmp2 = query_host[criterion_args->feature_indices[0]*query_width + i] - ref_host[criterion_args->feature_indices[0]*ref_width + j];
					criterion_args->dist_host[i+j*query_width] += -multiplier * tmp1 * tmp1 + ((criterion_val >= criterion_args->feature_criterion_values[0])?0:multiplier * tmp2 * tmp2);

					//DEBUGGING!!!!!!!!!!!!!!!
					//printf("i: %d \t j: %d \t criterion_args->dist_host[i+j*query_width]: %f \t tmp1: %f tmp2: %f\n", i, j, criterion_args->dist_host[i+j*query_width], tmp1, tmp2);
				}
			}
			//DEBUGGING!!!!!!!!!!!!!!!
			//printf("criterion_val >= criterion_args->feature_criterions[0]: %d criterion_val: %f criterion_args->feature_criterions[0]: %f \n", criterion_val >= criterion_args->feature_criterion_values[0], criterion_val, criterion_args->feature_criterion_values[0]);
		}
		/*cudaEventRecord(function_end, 0);
		cudaEventSynchronize(function_end);
		cudaEventElapsedTime(&elapsed_time, function_start, function_end);*/
		//printf("Elapsed time for fixing the values of dist_host: %s\n", get_elapsed(elapsed_time/1000));
#endif
	}
	
	
	/************************************************** ADDED BY FATEMEH */   
 
    // CUBLAS shutdown
    //cublasShutdown();		/* COMMENTED OUT BY FATEMEH (since we don't use CUBLAS) */

	/*cudaEventDestroy(function_start);   
	cudaEventDestroy(function_end); */
}



//-----------------------------------------------------------------------------------------------//
//                                Main Code                                                      //
//-----------------------------------------------------------------------------------------------//

int main(int argc, char *argv[]) 
{
    enum 
	{
		WRKLD_NAME = 0,		// "mysql"
		WRKLD_PATH,			// "Workload/winxp_mysql_server_I.tpcc"
		DATA_FLDR,			// "Malfease.2008-6.7.8.9.10.nosignals"
		DATA_FILE,			// "MalwareNames.Malfease.2008-6.7.8.9.10.success.txt"
		RESULTS_FLDR,		// "forward_feature_selection_fldr"
		K,					// The k used to find the "k neighborhood"
		SIGMA,				// The sigma value for the Gaussian kernel
		DATA_COUNT,			// The total number of data points
		FEATURE_COUNT,		// The total number of features
		USE_INITIAL,		// Whether or not to use an initial set of features
		INITIAL_FILE,		// The name of the file containing the initial set of features
		INITIAL_CRITERIONS,	// The name of the file containing the initial criterion values for each feature count
		RUN_FLOATING_SEARCH,// Whether to run *floating* forward search or regular forward search (with no backward steps)
		RUN_BACKWARD_SEARCH,// Whether to run backward search or forward search
		NUM_ARGS
	};

	// Input parameters
	char *wrkld_name, *wrkld_path, *data_fldr, *data_file, *results_fldr, *initial_file, *initial_criterions;
	int k, data_count, feature_count, use_initial, run_floating_search, run_backward_search;
	float sigma;

	int i, j, f, /*g,*/ s, feature_id, use_features_count, max_steps, count,
		first_round,		// Whether this is the first round of the feature selection
		feature_combos,		// The number of different feature combinations we would like to try	
		features_to_remove,	// The number of features to remove during a leap of the backward search
		*features_deleted,	// The features in order of their deletion for backward search
		*current_use_features,
		*use_features;	// An array representing which features to use (1 = use feature, 0 = don't use feature) 
		
	FILE *initial_file_fp, *initial_criterions_fp, *step_criterions_file_fp, *output_fp, *fp,
		*data_file_fp, *time_fp; // File pointers to the respective files
	//char *filename, *data_names_filename, *all_features_filename;	// The name of the final feature file
	Arg * runArgs[NUM_ARGS];

	float *data_matrix,		// Entire matrix of the data points with all the features (data points in the columns, features in the rows)	NOTE: This is the transpose of what is used in the CPU version of the code
		//**current_data_matrix,	// Matrix of the data points with the current set of features 
		//*weights,				// Need it for calculating the criterion function
		//*distance_matrix,		// NOTE: I moved this into calculate_LoKDR_criterion
		*means, *std_devs,
		value,
		criterion_value, best_criterion, 
		*best_criterion_per_feature_count;

	char line[100*MAX_FEATURE_COUNT], 
		command[2*MAX_FILE_NAME_SIZE],		// The "2*" is to accommodate for the full path information as well
		filename[MAX_FILE_NAME_SIZE], *token;

	float fraction_to_remove = 0.1;		// The fraction of features to remove during each round of the backward search

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// Variables for work on GPU
	size_t /*unsigned int*/ memory_total;	/* MODIFIED BY FATEMEH */
	size_t /*unsigned int*/ memory_free;	/* MODIFIED BY FATEMEH */
	cudaError_t  result;

	gpu_args *criterion_args;
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Init 
	srand(time(NULL));
	cudaPrintfInit();

	// Variables for duration evaluation
	cudaEvent_t start, stop, feature_selection_start;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&feature_selection_start);
	float elapsed_time;

	unsigned int size_of_float = sizeof(float);

	for (i = 0; i < NUM_ARGS; i++)
		runArgs[i] = arg_alloc();

	runArgs[WRKLD_NAME]->type = ARG_STRING;
	runArgs[WRKLD_NAME]->descriptor = "-wrkld_name";
	runArgs[WRKLD_PATH]->type = ARG_STRING;
	runArgs[WRKLD_PATH]->descriptor = "-wrkld_path";
	runArgs[DATA_FLDR]->type = ARG_STRING;
	runArgs[DATA_FLDR]->descriptor = "-data_fldr";
	runArgs[DATA_FILE]->type = ARG_STRING;
	runArgs[DATA_FILE]->descriptor = "-data_file";
	runArgs[RESULTS_FLDR]->type = ARG_STRING;
	runArgs[RESULTS_FLDR]->descriptor = "-results_fldr";
	runArgs[K]->type = ARG_INTEGER;
	runArgs[K]->descriptor = "-k";
	runArgs[SIGMA]->type = ARG_STRING;
	runArgs[SIGMA]->descriptor = "-sigma";
	runArgs[DATA_COUNT]->type = ARG_INTEGER;
	runArgs[DATA_COUNT]->descriptor = "-data_count";
	runArgs[FEATURE_COUNT]->type = ARG_INTEGER;
	runArgs[FEATURE_COUNT]->descriptor = "-feature_count";
	runArgs[USE_INITIAL]->type = ARG_INTEGER;
	runArgs[USE_INITIAL]->descriptor = "-use_initial";
	runArgs[INITIAL_FILE]->type = ARG_STRING;
	runArgs[INITIAL_FILE]->descriptor = "-initial_file";
	runArgs[INITIAL_CRITERIONS]->type = ARG_STRING;
	runArgs[INITIAL_CRITERIONS]->descriptor = "-initial_criterions";
	runArgs[RUN_FLOATING_SEARCH]->type = ARG_INTEGER;
	runArgs[RUN_FLOATING_SEARCH]->descriptor = "-run_floating_search";
	runArgs[RUN_BACKWARD_SEARCH]->type = ARG_INTEGER;
	runArgs[RUN_BACKWARD_SEARCH]->descriptor = "-run_backward_search";

	for (i = 0; i < NUM_ARGS; i++)
	{
		getArg(argc, argv, runArgs[i]);
		printArg(runArgs[i]);
		if ( !runArgs[i]->valid ) 
		{
			printf("Error... \n");
			printf("Missing argument: %s\n", runArgs[i]->descriptor);		
			fatalError("Example $ ./lokdr -wrkld_name cns -wrkld_path Workload/Microarray -data_fldr cns -data_file cns_data_labels.csv -results_fldr kernel_density_results -k 10 -sigma 1.0 -data_count 90 -feature_count 7129 -use_initial 1 -initial_file Workload/cns_features.txt -initial_criterions cns_criterions.txt -run_floating_search 1 -run_backward_search 0\n");
		}
	}
	printf("\n");

	wrkld_name = runArgs[WRKLD_NAME]->data._str;
	wrkld_path = runArgs[WRKLD_PATH]->data._str;
	data_fldr = runArgs[DATA_FLDR]->data._str;
	data_file = runArgs[DATA_FILE]->data._str;
	results_fldr = runArgs[RESULTS_FLDR]->data._str;
	k = runArgs[K]->data._int;
	sigma = atof(runArgs[SIGMA]->data._str);
	data_count = runArgs[DATA_COUNT]->data._int;
	feature_count = runArgs[FEATURE_COUNT]->data._int;
	use_initial = runArgs[USE_INITIAL]->data._int;
	initial_file = runArgs[INITIAL_FILE]->data._str;
	initial_criterions = runArgs[INITIAL_CRITERIONS]->data._str;
	run_floating_search = runArgs[RUN_FLOATING_SEARCH]->data._int;
	run_backward_search = runArgs[RUN_BACKWARD_SEARCH]->data._int;

	if (run_floating_search && run_backward_search)
		fatalError("Cannot run both forward floating search and backward search. Please choose either forward floating search, forward search, or backward search.\n"); // Backward floating search is not currently implemented

	current_use_features = (int *)malloc(feature_count * sizeof(int));			// The current set of features to use
	use_features = (int *)malloc(feature_count * feature_count * sizeof(int));	// The matrix of features to use (each row corresponds to a possible combination of features to use; there are feature_count rows because there are at most feature_count combinations of features to try in a round, e.g., in the first round of forward or backward search)
	best_criterion_per_feature_count = (float *)malloc(feature_count * size_of_float);
	//weights = (float *)malloc(feature_count * size_of_float);

	if (run_backward_search)
	{
		features_deleted = (int *)malloc(feature_count * sizeof(int));
	}

	// If it does not exist, create the folder that will contain the feature selection results
	sprintf(command, "[ -e %s%s%s%s ] || mkdir %s%s%s%s", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR);
	system(command);

	if (run_floating_search)
		max_steps = 1000;	// The maximum number of forward/backward steps
	else
	{
		if (run_backward_search)
		{
			max_steps = INT_MAX;	// For backward search, features will be removed one by one until the very last feature
			
			// Prepare the file containing the backward search features in the reverse order that they were removed
			sprintf(filename, "%s%s%s%s%s_k_%d_sigma_%.2lf_backward_search_features.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, k, sigma);
			if ( !(fp = fopen(filename, "w"))) 
			{
				fatalError("Can't open %s\n", filename);		
			}
		}
		else
		{
			max_steps = 120;	// For regular forward search, features will only be added so stop once max_steps features are added
		
			// Prepare the file containing the forward search features in the order that they were added
			sprintf(filename, "%s%s%s%s%s_k_%d_sigma_%.2lf_forward_search_features.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, k, sigma);
			if ( !(fp = fopen(filename, "w"))) 
			{
				fatalError("Can't open %s\n", filename);		
			}
		}
	}

	//// Initialize the weights array (for criterion calculation)
	//for(i = 0; i < feature_count; i++)
	//	weights[i] = 1;

	if (!use_initial)
	{
		// Initialize current_use_features array
		for (i = 0; i < feature_count; i++)
		{
			if (run_backward_search)
			{
				current_use_features[i] = 1;
				for (j = 0; j < feature_count; j++)
				{
					use_features[i * feature_count + j] = 1;	// I switched the place of "i" and "j" to improve cache performance (does not affect overall result since it will all be ones), i.e., rather than make the entire column 1's, I make the entire row 1's
				}
			}
			else
			{
				current_use_features[i] = 0;
				for (j = 0; j < feature_count; j++)
				{
					use_features[i * feature_count + j] = 0;	// I switched the place of "i" and "j" to improve cache performance (does not affect overall result since it will all be zeroes), i.e., rather than make the entire column 0's, I make the entire row 0's
				}
			}
		}
		if (run_backward_search)
		{
			use_features_count = feature_count;
			for (i = 0; i < feature_count; i++)
			{
				best_criterion_per_feature_count[i] = -1;
			}
		}
		else
			use_features_count = 0;

		best_criterion = -1;
	}
	else // If use_initial == 1, read in the initial set of features from the file initial_file
    {
		use_features_count = 0;
		if ( !(initial_file_fp = fopen(initial_file, "r"))) 
		{
			fatalError("Can't open %s\n", initial_file);		
		}
		while (fscanf(initial_file_fp, "%d", &feature_id) != EOF) 
		{  
			current_use_features[feature_id] = 1;
			for (i = 0; i < feature_count; i++)
				use_features[i*feature_count + feature_id] = 1;
			use_features_count++;
		}
		fclose(initial_file_fp);

		// Read in the criterion values
		i = 0;
		if ( !(initial_criterions_fp = fopen(initial_criterions, "r"))) 
		{
			fatalError("Can't open %s\n", initial_criterions);		
		}
		while (fscanf(initial_criterions_fp, "%lf", &criterion_value) != EOF) 
		{  
			best_criterion_per_feature_count[i] = criterion_value;
			i++;
		}
		best_criterion = criterion_value;
		fclose(initial_criterions_fp);
	}

	// Prepare the step_criterions.csv output file
	sprintf(filename, "%s%s%s%sstep_criterions.csv", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR);
	if ( !(step_criterions_file_fp = fopen(filename, "w"))) 
	{
		fatalError("Can't open %s\n", filename);		
	}

	// Read in the entire data matrix
	sprintf(filename, "%s%s%s%s%s", wrkld_path, PATH_SEPARATOR, data_fldr, PATH_SEPARATOR, data_file);
	if ( !(data_file_fp = fopen(filename, "r"))) 
	{
		fatalError("Can't open %s\n", filename);		
	}
	data_matrix = (float *)malloc((feature_count+1) * data_count * sizeof(float *)); // The label is in the last row
	means = (float *)malloc(feature_count * size_of_float);
	std_devs = (float *)malloc(feature_count * size_of_float);

	readLine(data_file_fp, line, 100*MAX_FEATURE_COUNT);

	// Initialize the mean and standard deviation vectors (use for z-score normalization)
	for (i = 0; i < feature_count; i++)
	{
		means[i] = 0;
		std_devs[i] = 0;
	}

	count = 0;
	// Read in the data matrix
	do
	{
		// Parse the feature values on each line
		f = 0;
		token = strtok(line, ",");
		while(token != NULL)
		{
			value = atof(token);
			data_matrix[f*data_count + count] = value;	// NOTE: Based on the way the k-NN computations work, the data points are in the columns and the features are in the rows (opposite of the configuration I used in the CPU version of the code)
			if (f < feature_count)
				means[f] += value;
			f++;
			
			// Get next token: 
			token = strtok(NULL, ",");
		}
		assert(f == feature_count+1);	// The last value read will be the label	
		count++;
	} while(readLine(data_file_fp, line, 100*MAX_FEATURE_COUNT) > 0);
	assert(count == data_count);
	fclose(data_file_fp);

	for (j = 0; j < feature_count; j++)
	{
		means[j] /= data_count;
		for (i = 0; i < data_count; i++)
			std_devs[j] += (data_matrix[j*data_count + i] - means[j]) * (data_matrix[j*data_count + i] - means[j]);
		std_devs[j] = sqrt(std_devs[j]/(data_count - 1)); 
		if (std_devs[j] == 0)
		{
			if (means[j] == 0)
				std_devs[j] = (float)1/(data_count * data_count * data_count);
			else std_devs[j] = means[j]/(data_count * data_count);	
		}
	}

	// Normalize the data matrix
	for (j = 0; j < feature_count; j++)
	{
		for (i = 0; i < data_count; i++)
		{
			data_matrix[j*data_count + i] = (data_matrix[j*data_count + i] - means[j])/std_devs[j];

			// DEBUGGING!!!!!!!!!!!!!!
			//printf("j: %d \t i: %d \t data_matrix[j*data_count + i]: %f\n", j , i, data_matrix[j*data_count + i]);
		}
	}

	// Free the resources for means and std_devs
	free(means);
	free(std_devs);

	cudaEventRecord(feature_selection_start, 0);		// Start the timer for the beginning of the feature selection
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Set up the variables ans memory for work on the GPU

	criterion_args = (gpu_args *) malloc(sizeof(gpu_args));	/* ADDED BY FATEMEH */

	// Prepare the vectors for criterion values and indices of the affected features (i.e., the features are either added or removed) 
	criterion_args->feature_criterion_values = (float *) malloc(feature_count * size_of_float);
	criterion_args->feature_indices = (int *)malloc(feature_count * sizeof(int));

	criterion_args->dist_host = (float *) malloc(data_count * /*k * */ data_count * size_of_float);	/* ADDED BY FATEMEH */ // The data_count * data_count is because we went to reuse the distance calculations
	criterion_args->kde_values = (float *) malloc(data_count * /*k * */ feature_count * size_of_float);	/* ADDED BY FATEMEH */ // The data_count is to store the KDE values for each point; the feature_count is because we find the KDE for at most feature_count combinations of features

	criterion_args->kde_sum_normal = (float *) malloc(feature_count * size_of_float);
	criterion_args->kde_sum_abnormal = (float *) malloc(feature_count * size_of_float);

	// Check if we can use texture memory for reference points		/* ADDED BY FATEMEH (from the CUDA version, for texture memory use) */
    criterion_args->use_texture = ( data_count*size_of_float<=MAX_TEXTURE_WIDTH_IN_BYTES && feature_count*size_of_float<=MAX_TEXTURE_HEIGHT_IN_BYTES );	
	//CHANGE BACK LATER!!!!!!!!!!!!%%%%%%%%%%%################
    
	
	
	
	
	
	
	
	// CUDA Initialisation
    cuInit(0);
	//cublasInit();		/* COMMENTED OUT BY FATEMEH (since we are not using CUBLAS) */
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    cuCtxDetach (cuContext);
	
    // Determine maximum number of query points that can be treated
	// In the following line, we find the maximum number of query points to evaluate by taking the amount of free memory we want to use and subtract the amount of memory for the reference set and divide the result by the amount of data needed for each query point which is the space for the point itself and its corresponding distance matrix (including at most feature_count combinations of features) 
	criterion_args->max_nb_query_treated = (size_t)( (unsigned int)memory_free * MAX_PART_OF_FREE_MEMORY_USED - (criterion_args->use_texture?0:size_of_float * data_count * feature_count) ) / ( size_of_float * (feature_count /*+ 1*/ + data_count * feature_count) );	/* MODIFIED BY FATEMEH (casting to "unsigned int" and I removed the "+ 1" since we are not storing the query norm ||query||, used in the CUBLAS version)*/	// NOTE: I decided to comment out "+1" because we don't use the label on the GPU
    criterion_args->max_nb_query_treated = min( data_count, (unsigned int)(criterion_args->max_nb_query_treated / BLOCK_DIM) * BLOCK_DIM );	/* MODIFIED BY FATEMEH (casting to "unsigned int") */
	
	//// Allocation of global memory for query points, ||query||, and for 2.R^T.Q
 //   result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_treated * size_of_float, (height + ref_width + 1));
 //   if (result){
 //       printErrorMessage(result, max_nb_query_treated * size_of_float * ( height + ref_width + 1 ) );
 //       return;
 //   }
 //   query_pitch = query_pitch_in_bytes/size_of_float;
	//query_norm  = query_dev  + height * query_pitch;
 //   dist_dev    = query_norm + query_pitch;
 //   
 //   // Allocation of global memory for reference points and ||query||
 //   result = cudaMallocPitch((void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height+1);
 //   if (result){
 //       printErrorMessage(result, ref_width * size_of_float * ( height+1 ));
 //       cudaFree(query_dev);
 //       return;
 //   }
 //   ref_pitch = ref_pitch_in_bytes / size_of_float;
 //   ref_norm  = ref_dev + height * ref_pitch;
 //   
 //   // Memory copy of ref_host in ref_dev
 //   result = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float, ref_width*size_of_float, height, cudaMemcpyHostToDevice);
 //   

	/* ADDED BY FATEMEH (from the CUDA version) */

	// Allocation of global memory for query points and for distances
    result = cudaMallocPitch( (void **) &(criterion_args->query_dev), &(criterion_args->query_pitch_in_bytes), criterion_args->max_nb_query_treated * size_of_float, feature_count /*+ 1*/ /*+ ref_width * feature_combinations*/);	/* MODIFIED BY FATEMEH (multiplied ref_width by feature_combinations since we are going to have "feature_combinations" number of distance matrices, we use "+ 1" because after the last feature is the label of the data point) */ // NOTE: I decided to comment out "+1" because we don't use the label on the GPU
    if (result){
        printErrorMessage(result, criterion_args->max_nb_query_treated*size_of_float*(feature_count/*+ref_width*/));
        return -1;
    }
    criterion_args->query_pitch = criterion_args->query_pitch_in_bytes/size_of_float;
	//dist_dev    = query_dev + height * query_pitch;

	// ADDED BY FATEMEH ///////////////////////////////////////////////////////////////////////////////
	/* (NOTE: This program currently supports using a constant k for all data points, to use the LOF notion of variable k-neighborhood size, change k to data_count and modify the insertion sort appropriately) */
	result = cudaMallocPitch( (void **) &(criterion_args->dist_dev), &(criterion_args->dist_pitch_in_bytes), criterion_args->max_nb_query_treated  * feature_count /*feature_combinations*/ * size_of_float, data_count);	/* MODIFIED BY FATEMEH (multiplied data_count by feature_combinations since we are going to have "feature_combinations" number of distance matrices, we use "+ 1" because after the last feature is the label of the data point) */ // NOTE: I decided to comment out "+1" because we don't use the label on the GPU
    if (result){
        printErrorMessage(result, criterion_args->max_nb_query_treated*feature_count*size_of_float*(/*height+*/data_count));
        return -1;
    }
	criterion_args->dist_pitch = criterion_args->dist_pitch_in_bytes/size_of_float;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    
	// Allocation of memory (global or texture) for reference points
    if (criterion_args->use_texture){

#ifdef PRINT_PROGRESS
		printf("\nUsing texture memory\n");		/* ADDED BY FATEMEH */
#endif
	
        // Allocation of texture memory
        cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float>();
        result = cudaMallocArray( &(criterion_args->ref_array), &channelDescA, data_count, feature_count );
        if (result){
            printErrorMessage(result, data_count*feature_count*size_of_float);
            cudaFree(criterion_args->query_dev);
            return -1;
        }
        cudaMemcpyToArray( criterion_args->ref_array, 0, 0, /*ref_host*/ data_matrix, data_count * feature_count * size_of_float, cudaMemcpyHostToDevice );
        
        // Set texture parameters and bind texture to array
        texA.addressMode[0] = cudaAddressModeClamp;
        texA.addressMode[1] = cudaAddressModeClamp;
        texA.filterMode     = cudaFilterModePoint;
        texA.normalized     = 0;
        cudaBindTextureToArray(texA, criterion_args->ref_array);
		
    }
    else{
	
		// Allocation of global memory
        result = cudaMallocPitch( (void **) &(criterion_args->ref_dev), &(criterion_args->ref_pitch_in_bytes), data_count * size_of_float, feature_count);
        if (result){
            printErrorMessage(result,  data_count*size_of_float*feature_count);
            cudaFree(criterion_args->query_dev);
            return -1;
        }
        criterion_args->ref_pitch = criterion_args->ref_pitch_in_bytes/size_of_float;
        cudaMemcpy2D(criterion_args->ref_dev, criterion_args->ref_pitch_in_bytes, /*ref_host*/ data_matrix, data_count*size_of_float,  data_count*size_of_float, feature_count, cudaMemcpyHostToDevice);
    }

	/* ADDED BY FATEMEH ************************************************/

	// Allocation of global memory for use_features_matrix
    result = cudaMallocPitch((void **) &(criterion_args->ufm_dev), &(criterion_args->ufm_pitch_in_bytes), feature_count * size_of_float, feature_count);
    if (result){
        printErrorMessage(result, feature_count * size_of_float * feature_count);
        cudaFree(criterion_args->query_dev);
        return -1;
    }
    criterion_args->ufm_pitch = criterion_args->ufm_pitch_in_bytes / size_of_float;
    
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// Prepare the file that will contain the time it took (from the start of execution) to add each feature
	sprintf(filename, "%s%s%s%sTIME_%s_k_%d_sigma_%.2lf_%s_search.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, k, sigma, run_backward_search?"backward":"forward");
	if ( !(time_fp = fopen(filename, "w"))) 
	{
		fatalError("Can't open %s\n", filename);		
	}
	
	for (s = 0; s < max_steps && use_features_count <= feature_count && (!run_backward_search || (run_backward_search && use_features_count > 1)); s++)
	{
		first_round = (s == 0);				// Determine if this is the first round of the feature search
		criterion_args->feature_criterion_values[0] = -1;

		if (!run_backward_search && use_features_count < feature_count)
		{

#ifdef PRINT_PROGRESS
			printf("Feature count: %d\n", use_features_count);
#endif
			// Try to add features
#ifdef PRINT_PROGRESS
			printf("\nAttempting to add a feature...\n");
#endif
			cudaEventRecord(start, 0);		// Start the timer

			// Set up every combination of adding one feature
			i = 0;
			for (f = 0; f < feature_count; f++)
			{
				if (current_use_features[f] == 0)
				{
					criterion_args->feature_indices[i] = f;
#ifdef LESS_COMPS_SOME_COPYS
					if (first_round)
#endif
						use_features[i*feature_count + f] = 1;
#ifdef LESS_COMPS_SOME_COPYS
					else
					{
						use_features[i*feature_count] = 1;		// Set the first column of the use_features matrix to +1 since we are adding features
						use_features[i*feature_count + 1] = f;		// Set the second column of the use_features matrix to the index of the feature
					}
#endif
					i++;
				}
			}
			feature_combos = i;

			//distance_matrix = (float *) malloc(data_count * feature_combos * sizeof(float));		
			
			// Calculate the criterion function
			calculate_LoKDR_criterion(first_round, best_criterion, data_matrix, data_count, data_matrix, data_count, use_features, feature_combos, feature_count, k, sigma, criterion_args);
	

			//DEBUGGING!!!!!!!!!
			/*for (i = 0; i < feature_combos; i++)
				printf("feature: %d criterion: %f\n", feature_indices[i], feature_criterion_values[i]);*/
			//cudaPrintfDisplay(stdout, true);

			//free(distance_matrix);

			// If the best criterion value found so far (in the last iteration) is not lower that of this iteration, stop trying to add features
			// Otherwise update the best criterion value found so far
			//if (best_criterion >= criterion_args->feature_criterion_values[0])
			//{

#ifdef PRINT_PROGRESS
			//	printf("Adding another feature has not improved the best criterion found so far. Feature search will now end.\n");
			//	printf("Best criterion value = %lf, Current criterion value = %lf.\n", best_criterion, criterion_args->feature_criterion_values[0]);
			//	break;
#endif
			//}
			//else
			//{
			  best_criterion = criterion_args->feature_criterion_values[0];
			//}

			// Make a note of the best criterion value for this feature count
			best_criterion_per_feature_count[use_features_count] = best_criterion;

			// Add the feature with the best criterion to the set of selected features
			current_use_features[criterion_args->feature_indices[0]] = 1;
			use_features_count++;

			//// Update use_features matrix to prepare for the next step (round) of the search	// NOTE: Not needed because use_feature_matrix will later be used in an incremental manner (the full version is needed only when full distances are calculated, which happens during chunk feature removal in backward search)
			//for (j = 0; j < feature_count; j++)
			//{
			//	for (i = 0; i < feature_count; i++)	
			//	{
			//		use_features[i*feature_count + j] = current_use_features[j];
			//	}
			//}

#ifdef PRINT_PROGRESS
			printf("In step %d, feature %d is added with a criterion value of %lf\n", s, criterion_args->feature_indices[0], criterion_args->feature_criterion_values[0]);
#endif
			fprintf(step_criterions_file_fp, "%d, %lf, Added %d, Feature count: %d\n", s, criterion_args->feature_criterion_values[0], criterion_args->feature_indices[0], use_features_count);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed_time, start, stop);

#ifdef PRINT_PROGRESS
			printf("Elapsed time to add the feature: %s\n", get_elapsed(elapsed_time/1000));
#endif
			cudaEventElapsedTime(&elapsed_time, feature_selection_start, stop);
			fprintf(time_fp, "%f\n", elapsed_time/1000); 

			if (!run_floating_search)
				fprintf(fp, "%d 1\n", criterion_args->feature_indices[0]);	// The 1 represents the weight

			// Output the current set of features
			sprintf(filename, "%s%s%s%s%s_k_%d_sigma_%.2lf_feature_count_%d.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, k, sigma, use_features_count);
			if ( !(output_fp = fopen(filename, "w"))) 
			{
				fatalError("Can't open %s\n", filename);		
			}
			for (f = 0; f < feature_count; f++)
			{
				if (use_features[f])
					fprintf(output_fp, "%d 1\n", f);	// The 1 represents the weight
			}
			fclose(output_fp);
		}

		if ( (run_floating_search && use_features_count > 2) || run_backward_search)
		{

#ifdef PRINT_PROGRESS
			printf("Feature count: %d\n", use_features_count);
#endif
			// Try removing features
#ifdef PRINT_PROGRESS
			printf("\nAttempting to remove a feature...\n");
#endif
			criterion_args->feature_criterion_values[0] = -1;	// This ensures the block after the following "do" does not get executed in the first pass
			do
			{
				first_round = !run_floating_search && (s == 0);		// Check to see if it is the first round (or step) of the feature search
				if (use_features_count > 1 && best_criterion_per_feature_count[use_features_count - 2] < criterion_args->feature_criterion_values[0])   // The "use_features_count - 2" is because the indexing starts at 0 and we want to see the best criterion value when we had use_features_count - 1 features (which is placed at best_criterion_per_feature_count[use_features_count - 2]
				{
					if (run_backward_search && use_features_count > 100)
						features_to_remove = (int)(fraction_to_remove*use_features_count);
					else features_to_remove = 1;
					if (features_to_remove < 1)		// Check that we don't ask for 0 features to be removed
						features_to_remove = 1;
					
					for (j = 0; j < features_to_remove; j++)
					{
						if (run_backward_search)
							features_deleted[s] = criterion_args->feature_indices[j];

						// Remove the feature with the best criterion from the set of selected features
						current_use_features[criterion_args->feature_indices[j]] = 0;
						use_features_count--;

						s++;	// Count this as taking a backward step in the feature space search (Note: For backward search, s represents the number of "leaps" that are taken, not single steps)			
#ifdef PRINT_PROGRESS
						printf("In step %d, feature %d is removed with a criterion value of %lf\n", s, criterion_args->feature_indices[j], criterion_args->feature_criterion_values[j]);
#endif
						fprintf(step_criterions_file_fp, "%d, %lf, Removed %d, Feature count: %d\n", s, criterion_args->feature_criterion_values[j], criterion_args->feature_indices[j], use_features_count);
						cudaEventRecord(stop, 0);
						cudaEventSynchronize(stop);
						cudaEventElapsedTime(&elapsed_time, start, stop);
#ifdef PRINT_PROGRESS
						if (j == features_to_remove - 1)
							printf("Elapsed time to remove the feature: %s\n", get_elapsed(elapsed_time/1000));
#endif
						cudaEventElapsedTime(&elapsed_time, feature_selection_start, stop);
						fprintf(time_fp, "%f\n", elapsed_time/1000); 
					}

					//// Update use_features matrix to prepare for the next step (round) of the search	// NOTE: Not needed because use_feature_matrix will later be used in an incremental manner (the full version is needed only when full distances are calculated, which happens during chunk feature removal in backward search)
					//for (j = 0; j < feature_count; j++)
					//{
					//	for (i = 0; i < feature_count; i++)	
					//	{
					//		use_features[i*feature_count + j] = current_use_features[j];
					//	}
					//}
#ifdef PRINT_PROGRESS
					printf("\n");				// Helps to distinguish when chunks of features are removed
#endif
					fprintf(step_criterions_file_fp, "\n");
					
					if (features_to_remove > 1)		// If we removed multiple features at time, recalculate the criterion for the new number of features
					{
						//// Update use_features matrix to prepare for the following call to calculate_LoKDR_criterion	// Note: Instead of this, we can call calculate_LoKDR_criterion with current_use_features
						//for (f = 0; f < feature_count; f++)
						//{
						//	if (current_use_features[f] == 1)
						//	{
						//		use_features[f] = 1;	
						//	}
						//	else // (current_use_features[f] == 0)
						//	{
						//		use_features[f] = 0;	
						//	}
						//}
						
						/*feature_combos = 1;*/

						// In the following, we set feature_combos to 1 because we want it to calculate all the pairwise distances again since a bunch of features are being removed (the secong argument is set as DBL_MAX but it is really ignored)
						calculate_LoKDR_criterion(first_round, DBL_MAX, data_matrix, data_count, data_matrix, data_count, current_use_features/*use_features*/, 1 /*feature_combos*/, feature_count, k, sigma, criterion_args);					
					}

					best_criterion = criterion_args->feature_criterion_values[0];
					best_criterion_per_feature_count[use_features_count - 1] = best_criterion;	

					// Output the current set of features
					sprintf(filename, "%s%s%s%s%s_k_%d_sigma_%.2lf_feature_count_%d.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, k, sigma, use_features_count);
					if ( !(output_fp = fopen(filename, "w"))) 
					{
						fatalError("Can't open %s\n", filename);		
					}
					for (f = 0; f < feature_count; f++)
					{
						if (current_use_features[f])
							fprintf(output_fp, "%d 1\n", f);	// The 1 represents the weight
					}
					fclose(output_fp);
				}
				if (use_features_count > 1)	// Used to be "> 2" before I added backward search (because we want to know the better of the first two in backward search, but we already know it in forward search)
				{
					// Remove each of the features one at a time to see if it improves the criterion value (for that number of features)
					
					cudaEventRecord(start, 0);		// Start the timer

					// Set up every combination of removing one feature
					i = 0;
					for (f = 0; f < feature_count; f++)
					{
						if (current_use_features[f] == 1)
						{
							criterion_args->feature_indices[i] = f;
#ifdef LESS_COMPS_SOME_COPYS
							if (first_round)
#endif
								use_features[i*feature_count + f] = 0;
#ifdef LESS_COMPS_SOME_COPYS
							else
							{
								use_features[i*feature_count] = -1;		// Set the first column of the use_features matrix to -1 since we are removing features
								use_features[i*feature_count + 1] = f;	// Set the second column of the use_features matrix to the index of the feature
							}
#endif
							i++;
						}
					}
					feature_combos = i;

					//distance_matrix = (float *) malloc(data_count * k * feature_combos * sizeof(float));		
					
					// Calculate the criterion function
					calculate_LoKDR_criterion(first_round, best_criterion_per_feature_count[use_features_count - 2], data_matrix, data_count, data_matrix, data_count, use_features, feature_combos, feature_count, k, sigma, criterion_args);
				}
			} while(use_features_count > 1 && best_criterion_per_feature_count[use_features_count - 2] < criterion_args->feature_criterion_values[0]);	// Used to be "> 2"

			//printf("Feature count: %d\n", use_features_count);

#ifdef PRINT_PROGRESS
			if (!run_backward_search && use_features_count > 2)
			{
				printf("Feature %d is not removed\n", criterion_args->feature_indices[0]);
				printf("best_criterion_per_feature_count[use_features_count - 2]: %lf\n", best_criterion_per_feature_count[use_features_count - 2]);
				printf("max_criterion_value: %lf\n", criterion_args->feature_criterion_values[0]);
			}
#endif
		}

		// Determine the last feature remaining
		if (run_backward_search)
		{
			for (f = 0; f < feature_count; f++)
			{
				if (current_use_features[f] == 1)
				{
					features_deleted[feature_count - 1] = f;

#ifdef PRINT_PROGRESS
					printf("In step %d, the last feature remaining is %d\n", s, f);
#endif
				}
			}

			// Output the features in order of their importance (the reverse order in which they were removed)
			for (f = 0; f < feature_count; f++)
			{
				fprintf(fp, "%d 1\n", features_deleted[feature_count - 1 - f]);	// The 1 represents the weight
			}
			
			break;	// Done with backward search, can exit the outmost loop
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, feature_selection_start, stop);
	fprintf(time_fp, "%f\n", elapsed_time/1000); 

//#ifdef PRINT_PROGRESS
	printf("\n\nTime it took to run feature selection: %s\n", get_elapsed(elapsed_time/1000));
//#endif

	fclose(time_fp);

	fclose(step_criterions_file_fp);
	if (!run_floating_search || run_backward_search)
		fclose(fp);

	if (!run_backward_search)
	{
		// Output the final set of features
		sprintf(filename, "%s%s%s%s%s%s_forward_search_features_k_%d_sigma_%.2lf.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, run_floating_search?"_floating":"", k, sigma);
		if ( !(output_fp = fopen(filename, "w"))) 
		{
			fatalError("Can't open %s\n", filename);		
		}
		for (f = 0; f < feature_count; f++)
		{
			if (current_use_features[f])
				fprintf(output_fp, "%d 1\n", f);	// The 1 represents the weight
		}
		fclose(output_fp);
	}
  
	// Output the best criterion values for each feature count
	sprintf(filename, "%s%s%s%s%s%s_best_criterions_k_%d_sigma_%.2lf.txt", wrkld_path, PATH_SEPARATOR, results_fldr, PATH_SEPARATOR, wrkld_name, run_floating_search?"_floating":"", k, sigma);
	if ( !(output_fp = fopen(filename, "w"))) 
	{
		fatalError("Can't open %s\n", filename);		
	}
	for (f = 0; f < use_features_count; f++)
	{
		fprintf(output_fp, "%lf\n", best_criterion_per_feature_count[f]);
	}
	fclose(output_fp);

	// Free allocated resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	for(i = 0; i < NUM_ARGS; i++)
		arg_free(runArgs[i]);

	free(data_matrix);
	free(current_use_features);
	free(use_features);
	//free(weights);
	free(best_criterion_per_feature_count);

	if (run_backward_search)
	{
		free(features_deleted);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Free the allocated resources for the GPU

	free(criterion_args->feature_indices);
	free(criterion_args->feature_criterion_values);

	free(criterion_args->kde_sum_normal);
	free(criterion_args->kde_sum_abnormal);
	free(criterion_args->dist_host);
	free(criterion_args->kde_values);

	// Free memory
    if (criterion_args->use_texture)
        cudaFreeArray(criterion_args->ref_array);
    else
        cudaFree(criterion_args->ref_dev);
    
	cudaFree(criterion_args->query_dev);
	cudaFree(criterion_args->dist_dev);		/* ADDED BY FATEMEH */
	cudaFree(criterion_args->ufm_dev);		/* ADDED BY FATEMEH */

	free(criterion_args);

	cudaEventDestroy(start); 
	cudaEventDestroy(stop); 
	cudaEventDestroy(feature_selection_start); 

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cudaPrintfEnd();

	return 0;

#undef MAX_FILE_NAME_SIZE
#undef MAX_FEATURE_COUNT
}

/****************************************** Calculates the elapsed time *****************************************/
// Source of the main parts of the code: http://forums.devshed.com/c-programming-42/calculating-user-time-and-elapsed-time-51843.html

const int MINUTE = 60;
const int HOUR   = 60 * 60;
const int DAY    = 3600 * 24;

char *get_elapsed(float sec)
{
    int days    =0;
    int hours   =0;
    int minutes =0;
    int seconds =0;

    char* dstr = (char *)malloc(50*sizeof(char));
    char* et   = (char *)malloc(50*sizeof(char));
	char *res = (char *)malloc(50*sizeof(char));

	days    = (int)sec / DAY;
	if (days)    sec -= DAY * days;
	hours   = (int)sec / HOUR;
	if (hours)   sec -= HOUR * hours;
	minutes = (int)sec / MINUTE;
    if (minutes) sec -= MINUTE * minutes;

    seconds = (int)sec;
    sprintf(et, "%02d:%02d:%02d", hours, minutes, seconds);
	sprintf(res, "%s", et);

    if (days) {
        sprintf(dstr, "%d Day%s, ", days, &"s"[days == 1]);
        strncat(dstr, et, strlen(et));
        sprintf(res, "%s", dstr);
    }
	
    free(dstr);
    free(et);
    return res;
}
