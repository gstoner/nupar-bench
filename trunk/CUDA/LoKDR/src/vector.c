#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "vector.h"



#define DATA_ARRAY_QUANTUM	10000
#define ROOT_QUANTUM		100

/* this function increases root array.
   if not allocated, it will allocate and set to NULL.
   if allocated, it will allocate a bigger one, copy the data array pointers and
	set the rest to zero.
   Modifies v->root, v->root_size!
   returns 1 - on success,
		   0 - on failure (can't allocate)
*/
static int root_increase(Vector *v) 
{
	// If root was never allocated
	if ( ! v->root ) 
	{
		assert( v->root_size == 0 );
		assert( v->vector_size == 0 );
		v->root_size = ROOT_QUANTUM;

		// root is an array of userData arrays
		if ( ! ( v->root = (userData**)malloc(sizeof(userData*) * v->root_size) ) )
			return 0;		

		memset(v->root, 0, sizeof(userData*) * v->root_size);
	}
	// root was allocated, need to reallocate
	else {
		assert( v->root_size );
		assert( v->vector_size );
		assert( v->vector_size == v->root_size * DATA_ARRAY_QUANTUM );
		assert( v->root[v->root_size - 1] != NULL );

		// realloc
		if ( ! ( v->root = (userData**)realloc(v->root, sizeof(userData*) * v->root_size * 2) ) )
			return 0;
		
		// reset second half of root..
		memset(&(v->root[v->root_size]), 0, sizeof(userData*) * v->root_size);
		
		// double root size
		v->root_size *= 2;
	}
	return 1;
}

/* this function simply allocates a data array.
   returns array if all o.k., NULL otherwise
*/
static userData* data_array_alloc(Vector *v) 
{
	userData *data_array;

	// alloc
	data_array = (userData*)malloc(sizeof(userData) * DATA_ARRAY_QUANTUM);
	
	if ( data_array ) {
		memset(data_array, 0, sizeof(userData) * DATA_ARRAY_QUANTUM);
	}

	return data_array;
}


/* constructs a new Vector.
   returns a handle to vector
   returns NULL if error (allocating memory) occured */
Vector* VectorConstruct(void)
{
	Vector *v;
	if ( ! ( v = (Vector*)malloc(sizeof(Vector)) ) )
		return NULL;

	v->root = NULL;
	v->root_size = 0;
	v->vector_size = 0;

	return v;
}


/* deletes/does:
	1. if function is supplied, execute function on data (used to delete) (
	2. deletes all elements
	3. deletes handle.
   */
void VectorDestruct(Vector *v, PfnVectorCallback Callback)
{
	unsigned int i = 0;
	assert(v);
	if ( Callback ) 
		VectorExecute(v, Callback);

	if ( v->root_size )
	{
		// delete all data arrays
		while ( v->root[i] && i < v->root_size ) {
			free ( v->root[i] );
			i ++;
		}
		// delete root
		free ( v->root );
	}

	// delete Vector
	free(v);
}

/* traverses vector, call Callback on each data item.	*/
void VectorExecute(Vector *v, PfnVectorCallback Callback)
{
	unsigned int i;

	assert(v);
	assert(Callback != NULL);

	for ( i = 0; i < v->vector_size; i++ ) {
		(*Callback)(VectorGet(v, i));
	}
}

/* returns data corresponding to i,
   data can not be NULL, it will return data */
void* VectorGet(Vector *v, unsigned int i) {	
	unsigned int root_index;
	unsigned int data_array_index;
	
	assert(v);
	assert(i < v->vector_size);

	root_index			= i / DATA_ARRAY_QUANTUM;
	data_array_index	= i % DATA_ARRAY_QUANTUM;
	
	assert(root_index < v->root_size);
	assert(data_array_index < DATA_ARRAY_QUANTUM);
	assert(v->root[root_index]);
	assert(v->root[root_index][data_array_index]);

	return v->root[root_index][data_array_index];
}


/* insert data at a newly element at the end of the vector, (InsertAtEnd)
   data is not copied - we save the pointer! 
   Note, for validation - data is not allowed to be NULL,
   return 1 on success
		  0 on failure (can't allocate) */
int VectorPush(Vector *v, void *data)
{
	int rc;
	unsigned int root_index;
	unsigned int data_array_index;

	assert(v);
	assert(data);

	root_index			= v->vector_size / DATA_ARRAY_QUANTUM;
	data_array_index	= v->vector_size % DATA_ARRAY_QUANTUM;
	
	// First, make sure root is allocated & big enough
	if ( root_index >= v->root_size ) 
	{
		rc = root_increase(v);
		if ( ! rc ) return 0; // failed to allocate
	}

	// now, root is large enough, if we are not pointing to a valid array, allocate one
	if ( v->root[root_index] == NULL ) {
		v->root[root_index] = data_array_alloc(v);
		if ( ! v->root[root_index] ) return 0; // failed to allocate
	}

	// now, we have everything allocated.
	// lets verify we access an invalid location and set the data
	assert(v->root[root_index][data_array_index] == NULL);
	v->root[root_index][data_array_index] = data;
	
	// increase vector size
	v->vector_size++;

	return 1;
}
