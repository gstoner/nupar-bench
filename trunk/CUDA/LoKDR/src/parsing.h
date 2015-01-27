#ifdef __cplusplus
  extern "C" {
#endif 

#ifndef __PARSING_H__
#define __PARSING_H__

#include <stdio.h>
#include "vector.h"

/*
 reads the next line (from fp) into str (including \n)
	str must be allocated, size is it's size.
 Function will add a \0 character to the end of line.
 returns:
	on success - number of bytes read > 0
	on eof     - 0
	on failure - < 0
		-1 line too long (str not big enough).
		-2 error occured
*/
int readLine(FILE *fp, char str[], int size);

/*
  Input:   filename
		   chomp (boolean)
				if 0 - line is not modified and '\n' is preserved
				if 1 - line ending (\n or \r\n are removed)
  Output:  a vector of strings.
			each string corresponds to 1 line in the file
  Does: Opens the give file. Assuming this is a text file, it
		will read each line into a newly allocated string and
		push it to a vector.

  On error, NULL is returned.
*/
Vector* fileToStringVector(char filename[], int chomp);


/*
  Input:   Vector of strings 
		   filename
		   addNewLine
				if 0 - string is written as is
				if 1 - a line ending is printed after string
  Output:  
  Does: Opens the give file for writing.
		Writes all strings into file.
		Does not modify vector.
*/
void stringVectorToFile(Vector* v, char filename[], int addNewLine);


#endif /* __PARSING_H__ */

#ifdef __cplusplus
  }
#endif
