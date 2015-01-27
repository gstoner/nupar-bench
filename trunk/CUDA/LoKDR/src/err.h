#ifdef __cplusplus
  extern "C" {
#endif 

#ifndef __ERR_H__
#define __ERR_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "os.h"

/* vc++ 2003 Doesn't support Variadic Macros, gcc does. */
#ifndef WIN32
#define fatalError(...) do { \
		printf("Error in (file %s :: %d)\n", __FILE__, __LINE__); \
		printf(__VA_ARGS__);	\
		exit(-1);	\
   } while (0);
#else 
INLINE fatalError(char *format, ...) {
	printf(format);	
	printf("\nMissing some information.. run on linux..\n");
	exit(-1);	
}
#endif 

#endif // __ERR_H__

#ifdef __cplusplus
  }
#endif
