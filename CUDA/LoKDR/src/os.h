#ifdef __cplusplus
  extern "C" {
#endif 

#ifndef __OS_H__
#define __OS_H__

#if defined  WIN32
  #define INLINE __inline
#elif defined  __GNUC__ || __LP64__ || __CYGWIN32__ 
  #define INLINE static inline
#else
  #error "Which OS/compiler are we? no appropriate define."
#endif


#if defined  WIN32
	// Can't find these definitions.
  #define STDIN_FILENO 0
  #define STDOUT_FILENO 1
  #define STDERR_FILENO 2

	// fdopen is depracated
  #define fdopen _fdopen
	// strdup is depracated
  #define strdup _strdup
#elif defined  __GNUC__ || __LP64__ || __CYGWIN32__ 
  #include <unistd.h> 
#else
  #error "Which OS are we? no appropriate define."
#endif

#endif // __OS_H__

#ifdef __cplusplus
  }
#endif
