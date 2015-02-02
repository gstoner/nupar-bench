/*
 * -- NUPAR: A Benchmark Suite for Modern GPU Architectures
 *    NUPAR - 2 December 2014
 *    Fanny Nina-Paravecino
 *    Northeastern University
 *    NUCAR Research Laboratory
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the Northeastern U.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */
/*
 * Include files
 */
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include "image.h"
#include "misc.h"
#include "accl.h"
#include <omp.h>
#include <limits.h>
#define MAX_LABELS 262144
#define BUF_SIZE 256

class errorHandler { };
using namespace std;
/*
 * ---------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------
 */
double getWallTime();
double getCpuTime();
void pgmRead(ifstream &file, char *buf);
image<uchar> *loadPGM(const char *name);
image<int> *imageUcharToInt(image<uchar> *input);
void savePGM(image<rgb> *im, const char *name);
void acclSerial(image<int> *imInt, int *spans, int *components,
                const int rows, const int cols, image<rgb> *output);
/*
 * RGB generation colors randomly
 */
rgb randomRgb()
{
    rgb c;

    c.r = (uchar)rand();
    c.g = (uchar)rand();
    c.b = (uchar)rand();
    return c;
}

/*
 * getWallTime: Compute timing of execution including I/O
 */
double getWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        printf("Error getting time\n");
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

/*
 * getCpuTime: Compute timing of execution using Clocks function from C++
 */
double getCpuTime()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

/*
 * acclSerial: ACCL computes CCL using sequential implementation
 * Parameters:
 * - imInt:         int*
 *                  Image matrix with integer values per pixel
 * - spans:         int*
 *                  Spans matrix stores the intermediate values of spans per row
 *                  using two elements per span (start position, end position)
 * - components:    int*
 *                  Components matrix stores labels per span
 * - rows:          const int
 *                  number of rows in the original image
 * - cols:          const int
 *                  number of columns in the original image
 * - output:        image<rgb>
 *                  image output with the respective color segment assigned to each pixel
 */
void acclSerial(int *imInt, int *spans, int *components, const int rows,
                const int cols, image<rgb> *output)
{
    const int width = cols;
    const int height = rows;
    const int rowsSpans = rows;
    const int colsSpans = ((cols+2-1)/2)*2; //ceil(cols/2)*2
    const int spansSize = colsSpans*rowsSpans;
    const int componentsSize = (colsSpans/2)*rowsSpans;        
    int colsComponents = colsSpans/2;
    memset(spans, -1, spansSize*sizeof(int));
    memset(components, -1, componentsSize*sizeof(int));

    /*
     * Find Spans
     */
    double wall0 = getWallTime();
    double cpu0  = getCpuTime();
    for(int i=0; i<rows-1; i++)
    {
        int current =-1;
        bool flagFirst = true;
        int indexOut = 0;
        int indexComp = 0;
        int comp = i*colsComponents;
        for (int j = 0; j < cols; j++)
        {
            if(flagFirst && imInt[i*cols+j]> 0)
            {
                current = imInt[i*cols+j];
                spans[i*colsSpans+indexOut] = j;
                indexOut++;
                flagFirst = false;
            }
            if (!flagFirst && imInt[i*cols+j] != current)
            {
                spans[i*colsSpans+indexOut] = j-1;
                indexOut++;
                flagFirst = true;                
                components[i*colsComponents+indexComp] = comp;  /*Add respective label*/
                indexComp++;
                comp++;
            }
        }
        if (!flagFirst)
        {
            spans[i*colsSpans+indexOut] = cols - 1;
            /*Add the respective label*/
            components[i*colsComponents+indexComp] = comp;
        }
    }

    /*
     * Merge Spans
     */
    int label = -1;
    int startX, endX, newStartX, newEndX;
    for (int i = 0; i < rowsSpans-1; i++) /*compute until penultimate row, since we need the below row to compare*/
    {
        for (int j=0; j < colsSpans-1 && spans[i*colsSpans+j] >=0; j=j+2) /*verify if there is a Span available*/
        {
            startX = spans[i*colsSpans+j];
            endX = spans[i*colsSpans+j+1];
            int newI = i+1; /*line below*/
            for (int k=0; k<colsSpans-1 && spans[newI*colsSpans+k] >=0; k=k+2) /*verify if there is a New Span available*/
            {
                newStartX = spans[newI*colsSpans+k];
                newEndX = spans[newI*colsSpans+k+1];
                if (startX <= newEndX && endX >= newStartX) /*Merge components*/
                {
                    label = components[i*(colsSpans/2)+(j/2)];          /*choose the startSpan label*/
                    for (int p=0; p<=i+1; p++)                          /*relabel*/
                    {
                        for(int q=0; q<colsSpans/2; q++)
                        {
                            if(components[p*(colsSpans/2)+q]==components[newI*(colsSpans/2)+(k/2)])
                            {
                                components[p*(colsSpans/2)+q] = label;
                            }
                        }
                    }
                }
            }
        }
    }

    double wall1 = getWallTime();
    double cpu1  = getCpuTime();
    cout << "Time Performance: ACCL serial" << endl;
    cout << "\tWall Time = " << (wall1 - wall0)*1000 << " ms" << endl;
    cout << "\tCPU Time  = " << (cpu1  - cpu0)*1000  << " ms" << endl;

    /*
     * Convert to a labeled image matrix
     */
    rgb *colors = new rgb[width*height];
    
    for (int index = 0; index < width*height; index++)
        colors[index] = randomRgb();
    
    for(int i=0; i<rowsSpans; i++)
    {
        for(int j=0; j<colsSpans ; j=j+2)
        {
            startX = spans[i*colsSpans+j];
            if(startX>=0)
            {
                endX = spans[i*colsSpans+j+1];
                for(int k=startX; k <=endX; k++)
                {
                    imRef(output, k, i)= colors[components[i*(colsSpans/2)+(j/2)]];
                }
            }
        }
    }
    savePGM(output, "Data/out1.pgm");
    delete [] colors;
}

/*
 * pgmRead: read a pgm image file
 * Parameters:
 * - file:  ifstream
 *          path of the pgm image file
 * - buf:   char*
 *          buffer where information will be allocated
 */
void pgmRead(ifstream &file, char *buf)
{
    char doc[BUF_SIZE];
    char c;

    file >> c;
    while (c == '#')
    {
        file.getline(doc, BUF_SIZE);
        file >> c;
    }
    file.putback(c);

    file.width(BUF_SIZE);
    file >> buf;
    file.ignore();
}

/*
 * loadPGM: load pgm file and return it in a image<uchar> structure
 * Parameters:
 * - name:  const char*
 *          path of the pgm image file
 * Return:
 * - image<uchar>: image loaded in an uchar structure
 */
image<uchar> *loadPGM(const char *name)
{
    char buf[BUF_SIZE];

    /*
     * read header
     */
    std::ifstream file(name, std::ios::in | std::ios::binary);
    pgmRead(file, buf);
    if (strncmp(buf, "P5", 2))
    throw errorHandler();

    pgmRead(file, buf);
    int width = atoi(buf);
    pgmRead(file, buf);
    int height = atoi(buf);

    pgmRead(file, buf);
    if (atoi(buf) > UCHAR_MAX)
    throw errorHandler();

    /* read data */
    image<uchar> *im = new image<uchar>(width, height);
    file.read((char *)imPtr(im, 0, 0), width * height * sizeof(uchar));
    return im;
}

/*
 * savePGM: save pgm file
 * Parameters:
 * - im:    image<rgb>
 *          image in rgb colors to save the final output image
 * - name:  const char*
 *          path for the image output file
 */
void savePGM(image<rgb> *im, const char *name)
{
    int width = im->width();
    int height = im->height();
    std::ofstream file(name, std::ios::out | std::ios::binary);

    file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
    file.write((char *)imPtr(im, 0, 0), width * height * sizeof(rgb));
}

/*
 * imageUcharToInt: convert image from uchar to integer
 * Parameters:
 * - input: image<uchar>
 *          image in uchar to convert to integer values
 * Return:
 * - image<int>: image with integer values
 */
image<int> *imageUcharToInt(image<uchar> *input)
{
    int width = input->width();
    int height = input->height();
    image<int> *output = new image<int>(width, height, false);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            imRef(output, x, y) = imRef(input, x, y);
        }
    }
    return output;
}

int main()
{
    cout<<"Accelerated Connected Component Labeling" << endl;
    cout<<"========================================" << endl;
    cout<<"Loading input image..." << endl;
    image<uchar> *input = loadPGM("Data/8Frames.pgm");
    const int width = input->width();
    const int height = input->height();

    /*
     * Declaration of Variables
     */
    image<int> *imInt = new image<int>(width, height);
    image<rgb> *output1 = new image<rgb>(width, height);
    image<rgb> *output2 = new image<rgb>(width, height);
    imInt = imageUcharToInt(input);

    const uint nFrames= 8;
    const int rows = nFrames*512;
    const int cols = 512;
    const int imageSize = rows*cols;
    int *image = new int[imageSize];
    memcpy(image, imInt->data, rows * cols * sizeof(int));

    /*
     * Buffers
     */
    const int colsSpans = ((cols+2-1)/2)*2; /*ceil(cols/2)*2*/
    const int spansSize = colsSpans*rows;
    const int componentsSize = (colsSpans/2)*rows;
    int *spans= new int[spansSize];
    int *components = new int[componentsSize];

    /*
     * Initialize
     */
    memset(spans, -1, spansSize*sizeof(int));
    memset(components, -1, componentsSize*sizeof(int));

    /*
     * CUDA
     */
    acclCuda(spans, components, image, nFrames, rows, cols);

    /*
     * Print output image
     */
    rgb *colors = new rgb[width*height];
    int startX, endX;
    for (int index = 0; index < rows*cols; index++)
        colors[index] = randomRgb();

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<colsSpans; j=j+2)
        {
            startX = spans[i*colsSpans+j];
            if(startX>=0)
            {
                endX = spans[i*colsSpans+j+1];
                for(int k=startX; k <=endX; k++)
                {
                        if (components[i*(colsSpans/2)+(j/2)] != -1)
                        {
                            imRef(output2, k, i)= colors[components[i*(colsSpans/2)+(j/2)]];
                        }
                        else
                            printf("Error some spans weren't labeled\n");
                }
            }
        }
    }

    /*
     * Free memory
     */
    delete [] colors;
    savePGM(output2, "Data/out2.pgm");

    /*---------------- SERIAL --------------------*/
    int *spansSerial= new int[spansSize];
    acclSerial(image, spansSerial, components, rows, cols, output1);
    printf("Segmentation ended.\n");
    return 0;
}
