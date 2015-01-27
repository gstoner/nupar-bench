#ifdef __cplusplus
  extern "C" {
#endif 

#ifndef __ARGS_H__
#define __ARGS_H__

#define MAX_OPT_LENGTH 2048

typedef enum _argType {
	NULL_ARG_TYPE = 0,
	ARG_BOOL,       /* no argument.. */
	ARG_INTEGER,    /* integer argument */
	ARG_STRING      /* string argument */
} ArgType;


typedef struct _arg {
	/* user can update */
	/* identifier i.e. "-inputFileName" */
	char *descriptor;	
	/* type of argument */
	ArgType type;

	/* don't touch - just read ! */
	/* a pointer which will be filled with the data (int*, char*) */
	int valid;
	union {
		int _int;
		char _str[MAX_OPT_LENGTH];
	} data;
} Arg;



/* allocates a new Arg. Initializes to 0.
   Returns new arg on success or
			Null of failure (to allocate) */
Arg* arg_alloc(void);
/* free pointer */
void arg_free(Arg *a);




/* A pointer to your Arg struct, i do not copy it.
   make sure this Arg was previously allocated. 
   you are responsible to set up:
	descriptor and 
	type 
   correctly
   I will only fill in the data if needed. */
void getArg(int argc, char *argv[], Arg *arg);

/* A pointer to your Arg  struct.
	Just printing the descriptor and argument according to it's type */
void printArg(Arg *arg);



/* following function are just for convenience.. 
	input: argc, argv and descriptor (i.e. "-file")
			value - an allocated array. 
			value_size - the size of the array.
    does: if the descriptor is found in the arguments, copy the value into 
		value array.		  
		  if the value_size is too small i'll generate a fatal error.		  
	return: 1 success, found something.
			0 failure, didn't find anything.			
*/
int findStringArg(int argc, char *argv[], char *descriptor, char *value, unsigned int value_size);


/* following function are just for convenience.. 
	input: argc, argv and descriptor (i.e. "-file")
			value - a pointer to an integer 
    does: if the descriptor is found in the arguments, copy the value into 
		the integer.		  
	return: 1 success, found something.
			0 failure, didn't find anything.			
*/
int findIntegerArg(int argc, char *argv[], char *descriptor, int *value);

/* following function are just for convenience.. 
	input: argc, argv and descriptor (i.e. "-file")
    does: if the descriptor is found in the arguments
	return: 1 success, found something.
			0 failure, didn't find anything.			
*/
int findBooleanArg(int argc, char *argv[], char *descriptor);




#endif /* __ARGS_H__ */

#ifdef __cplusplus
  }
#endif
