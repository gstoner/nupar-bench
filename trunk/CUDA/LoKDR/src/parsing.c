#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "parsing.h"
#include "err.h"

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
int readLine(FILE *fp, char str[], int size)
{
	int c;
	int i = 0;

	do { 
		c = fgetc(fp);
			/* line too long (making sure there's space for '\0') */
		if ( i+1 == size ) return -1;
			/* EOF */
		if ( c == EOF ) return 0;
			/* error */ 
		if ( c < 0 ) {
		  //int error = ferror(fp);
			perror("Error reading file\n");
			return -2;
		}
		/* all is well */
		str[i++] = c;
		
	} while ( c != '\n' );
	str[i++] = '\0';
	
	return i;
}



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
Vector* fileToStringVector(char filename[], int chomp)
{
#define MAX_LINE_SIZE 4096
	char str[MAX_LINE_SIZE];
	FILE *fp;
	Vector *v;
	int rc;

	assert(filename);

	if ( ! ( fp = fopen(filename, "r") ) ) {
		fatalError("Can't open file %s\n", filename);		
		return NULL;
	}

	v = VectorConstruct();
	
	while ( ( rc = readLine(fp, str, MAX_LINE_SIZE) ) )
	{
		char *newStr;
		// First, remove new line. replace \n or \r\n with \0		
		assert(strlen(str) + 1 == rc); // Add 1 for \0

		// Second, chomp if requested
		if ( chomp ) 
		{
			char *newline;
			if ( ( newline = strstr(str, "\r\n") ) ) {
				*newline = '\0';
			}
			else if ( ( newline = strstr(str, "\n") ) ) {
				*newline = '\0';
			}
			else {
				fatalError("No new line at end of line?\n");
			}
		}

		// Third, allocate and copy string
		if ( ! ( newStr = strdup(str) ) ) {
			VectorDestruct(v, VectorfreeDataCallback);
			fatalError("fileToStringVector:: Can't duplicate string \n");
		}

		// Last, push into vector.
		if ( ! VectorPush(v, (void*)newStr ) ) {
			VectorDestruct(v, VectorfreeDataCallback);
			fatalError("fileToStringVector:: Can't push string into vector\n");
		}
	}
	if ( rc == -1 ) {
		VectorDestruct(v, VectorfreeDataCallback);
		fatalError("fileToStringVector:: Line too long, increase MAX_LINE_SIZE\n");
	}
	if ( rc == -2 ) {
		VectorDestruct(v, VectorfreeDataCallback);
		fatalError("fileToStringVector:: Error reading file\n");
	}
	assert(rc == 0);

	return v;
#undef MAX_LINE_SIZE

}



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
void stringVectorToFile(Vector* v, char filename[], int addNewLine)
{
	FILE *fp;
	unsigned int i, size;


	assert(v);
	assert(filename);

	if ( ! ( fp = fopen(filename, "w") ) ) {
		fatalError("Can't open file %s\n", filename);		
		return;
	}

	size = VectorSize(v);
	for (i=0; i<size; i++)
	{
		char* str = (char*)VectorGet(v, i);
		fprintf(fp, "%s", str);
		if ( addNewLine ) {
			fprintf(fp,"\n");
		}
	}

	fclose(fp);
	return;
}


