#ifndef _LOKDR_H_
#define _LOKDR_H_

typedef struct gpu_arguments{	// A structure that contains the GPU arguments used in the calculate_LoKDR_criterion function
	float		*feature_criterion_values;	// The criterion values of the feature combinations
	int			*feature_indices;			// The indices of the features that are added/removed (a (+) signifies addition of the feature and (-) signifies its removal)
	float       *dist_dev;
	float       *query_dev;
	float       *ref_dev;
	//float        *query_norm;			/* COMMENTED OUT BY FATEMEH (since we are not using CUBLAS) */
	//float        *ref_norm;			/* COMMENTED OUT BY FATEMEH (since we are not using CUBLAS) */
	int			*ufm_dev;				// The use_feature_matrix on the device		/* ADDED BY FATEMEH */
	cudaArray   *ref_array;			/* ADDED BY FATEMEH (from the CUDA version, for texture memory use) */
	size_t      query_pitch;
	size_t      query_pitch_in_bytes;
	size_t      dist_pitch;			/* ADDED BY FATEMEH */
	size_t      dist_pitch_in_bytes;	/* ADDED BY FATEMEH */
	size_t      ref_pitch;
	size_t      ref_pitch_in_bytes;
	size_t      ufm_pitch;				// The use_feature_matrix pitch				/* ADDED BY FATEMEH */
	size_t      ufm_pitch_in_bytes;	// The use_feature_matrix pitch in bytes	/* ADDED BY FATEMEH */
	size_t      max_nb_query_treated;
	size_t      actual_nb_query_width;
	float		*kde_sum_normal, *kde_sum_abnormal;			/* ADDED BY FATEMEH */
	float		*dist_host;		// size: ref_width x query_width
	float		*kde_values;	// size: query_width x feature_count (to accommodate all of the feature combinations)
	unsigned int use_texture;
	unsigned int run_floating_search;
	float       *sorted_dist_dev;
	} gpu_args; 

#endif	// _LOKDR_H_
