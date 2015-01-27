#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "args.h"
#include "err.h"



/* allocates a new Arg.
   initializes to 0.
   Returns new arg on success or
			Null of failure (to allocate) */
Arg* arg_alloc(void) {
	Arg *a;
	if ( ! ( a = (Arg*)malloc(sizeof(Arg)) ) )
		return NULL;

	/* initialize everything to NULL, 0*/
	memset(a, 0, sizeof(Arg));
	return a;
}

/* free pointer */
void arg_free(Arg *a) {
	assert(a);
	free(a);
}


/* A pointer to your Arg struct, i do not copy it.
   make sure this Arg was previously allocated. 
   you are responsible to correctly set up:
	descriptor and 
	type 
   
   I will only fill in the data and valid if needed. 
   arg->valid will be set to 1 if we found something.
              otherwise it will be 0.
   arg->data will be set up according to the type:
	   	BOOL    no argument..  valid set to 1
		INTEGER -> arg->data._int			  take next argument as integer 
		STRING  -> arg->data._str		      take next argument as string
   
   basically, loop over argv, 
	   find if argv[x] == arg->descriptor
	   then according to arg->type
		   fill up the data

*/
void getArg(int argc, char *argv[], Arg *arg) 
{
	int i = 0;
	
	assert(arg->type != NULL_ARG_TYPE);
	assert(arg->descriptor != NULL);
	assert(arg->valid == 0);
	assert(arg->data._int  == 0);
	assert(strlen(arg->data._str) == 0);

	while ( i < argc ) {
		if ( 0 == strcmp(arg->descriptor, argv[i]) ) {
			/* found a match */
			switch (arg->type) {
				case ARG_BOOL:				/* no argument.. */
					break;
				case ARG_INTEGER:           /* integer argument */
					if ( i + 1 >= argc )
						fatalError("expect argument after %s", argv[i]);
					if ( 1 != sscanf(argv[i+1], "%d", &(arg->data._int)) ) 
						fatalError("expect integer type argument after %s", argv[i]);
					i++;
					break;
				case ARG_STRING:            /* string argument */
					if ( i + 1 >= argc )
						fatalError("expect argument after %s", argv[i]);
					assert( strlen(argv[i+1]) < MAX_OPT_LENGTH );
					if ( 1 != sscanf(argv[i+1], "%s", arg->data._str) )
						fatalError("expect string type argument after %s", argv[i]);
					i++;
					break;
				default:
					fatalError("What arg type? shoudn't be here..");
			};
			/* done looking.. */
			arg->valid = 1;
			break;
		}
		i++;
	}
}

/* A pointer to your Arg  struct.
	Just printing the descriptor and argument according to it's type */
void printArg(Arg *arg)
{
	assert(arg->type != NULL_ARG_TYPE);
	assert(arg->descriptor != NULL);

	switch (arg->type) {
		case ARG_BOOL:				/* no argument.. */
			printf("\t%s -> %s\n",arg->descriptor, 	(arg->valid != 0 ? "True" : "False"));
			break;
		case ARG_INTEGER:           /* integer argument */
			if ( ! arg->valid )
				printf("\t%s -> INVALID\n", arg->descriptor);
			else
				printf("\t%s -> %d\n",arg->descriptor, arg->data._int);
			break;
		case ARG_STRING:            /* string argument */
			if ( ! arg->valid )
				printf("\t%s -> INVALID\n", arg->descriptor);
			else
				printf("\t%s -> %s\n",arg->descriptor, arg->data._str);
			break;
		default:
			fatalError("What arg type? shoudn't be here..");
	};
}


int findStringArg(int argc, char *argv[], char *descriptor, char *value, unsigned int value_size)
{
	Arg *argp;
	argp = arg_alloc();
	
	/* settup the argument: */
	argp->type = ARG_STRING;
	argp->descriptor = descriptor;
	
	getArg(argc, argv, argp);
	
	if ( ! argp->valid ) return 0;

	/* copy argument back... */
	if ( strlen(argp->data._str) >= value_size )
		fatalError("findStringArg:: can't copy back to value, not enough space allocated.\n");
	strcpy(value, argp->data._str);

	/* free argument.. */
	arg_free(argp);
	return 1;
}

int findIntegerArg(int argc, char *argv[], char *descriptor, int *value)
{
	Arg *argp;
	argp = arg_alloc();
	
	/* settup the argument: */
	argp->type = ARG_INTEGER;
	argp->descriptor = descriptor;
	
	getArg(argc, argv, argp);
		
	if ( ! argp->valid ) return 0;

	/* copy argument back... */
	*value = argp->data._int;

	/* free argument.. */
	arg_free(argp);
	return 1;
}

int findBooleanArg(int argc, char *argv[], char *descriptor)
{
	Arg *argp;
	argp = arg_alloc();
	
	/* settup the argument: */
	argp->type = ARG_BOOL;
	argp->descriptor = descriptor;
	
	getArg(argc, argv, argp);

	if ( ! argp->valid ) return 0;

	/* free argument.. */
	arg_free(argp);
	return 1;
}
