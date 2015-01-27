#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <assert.h>
#include <stdlib.h>

#include "os.h"

/* A Vector is simply a group of arrays which actually hold the data.
   The group of arrays is pointed by another array - the root.
   (it can be a linked list or a hash table but i think another array is a little
	more efficient.)
*/

typedef void*		userData;


/* linked list object */
typedef struct _vector {
	unsigned int vector_size; // the number of valid data items.
	userData **root;		  // an array of data arrays.
	unsigned int root_size;	  // the size root. 
} Vector;

typedef void (*PfnVectorCallback)(void *data);

/* Define an example and usefull callback */
INLINE void VectorfreeDataCallback(void *data){	free(data);	}







/* constructs a new Vector.
   returns a handle to vector
   returns NULL if error (allocating memory) occured */
Vector* VectorConstruct(void);
/* deletes/does:
	1. if function is supplied, execute function on data (used to delete) (
	2. deletes all elements
	3. deletes handle.
   */
void VectorDestruct(Vector *v, PfnVectorCallback Callback);
/* traverses vector, call Callback on each data item.	*/
void VectorExecute(Vector *v, PfnVectorCallback Callback);

INLINE unsigned int VectorSize(Vector *v) {
	assert(v);
	return ( v->vector_size );
}

/* returns data corresponding to i,
   data can not be NULL, it will return data */
void* VectorGet(Vector *v, unsigned int i);

/* insert data at a newly element at the end of the vector, (InsertAtEnd)
   data is not copied - we save the pointer! 
   Note, for validation - data is not allowed to be NULL,
   return 1 on success
		  0 on failure (can't allocate) */
int VectorPush(Vector *v, void *data);

/* returns:
	0 if Vector has one or more elements
	1 if Vector has no elements */
INLINE int VectorIsEmpty(Vector *v) {
	return ( 0 == VectorSize(v) );
}



#endif /* __VECTOR_H__ */
