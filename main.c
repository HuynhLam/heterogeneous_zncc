/******************************************************************************
 * FILENAME :        main.c
 *
 * DESCRIPTION :
 *       OpenCL-implementation of the Zero-mean Normalized Cross Correlation(ZNCC)
 *       + Decode two 32-bit RGBA images                                (running on host-code)
 *       + Resize these images by 1/16 (from 2940x2016 to 735x504)
 *       + Transform these images to greyscale images
 *       + Implement ZNCC on these image with changeable window size, output of
 *         ZNCC is disparity map.
 *       + Cross check two output disparity maps
 *       + Occlusion filling one output disparity map from cross check  (running on host-code)
 *       + Normalize the disparity map to 0..255                        (running on host-code)
 *         (case after downscaled, MAXDISP is 64)
 *       + Output the result image to "depthmap.png"                    (running on host-code)
 *
 *
 * NOTES :
 *       + Make use of the lodepng lib: http://lodev.org/lodepng/
 *
 * AUTHOR :    Lam Huynh        START DATE :    10 March 2017
 *
 * MAJOR CHANGES : (checkout git for details)
 * NO    VERSION DATE          DETAIL
 * 01    0.1.1   10Mar17       Respawned host-code from C-implementation
 * 02    0.1.2   17Mar17       resize.cl corrected
 * 03    0.1.3   21Mar17       zncc.cl corrected
 * 04    0.1.4   26Mar17       cross_check.cl corrected
 * 05    0.1.5   31Mar17       occlusion_filling.cl worked
 * 06    0.1.5   05Apr17       normalize.cl worked
 * 07    0.2.1   10Apr17       remove occlusion_filling.cl & normalize.cl (got stuck in optimize)
 * 08    0.2.2   20Apr17       change host-code
 * 08    0.2.3   05May17       Final host-code
 *
 ******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "lodepng.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


const int DOWNSCALE         = 4;    // downscale 4x4 = 16 times
const uint32_t HALFWINSIZEX = 8;    // Window size on X-axis (width)
const uint32_t HALFWINSIZEY = 15;   // Window size on Y-axis (height)
const int THRESHOLD         = 2;    // Threshold for cross-checkings
const uint32_t WINSIZEAREA  = 527;  // (HALFWINSIZEX x 2 + 1) x (HALFWINSIZEX x 2 + 1) = WINSIZEAREA

int MAXDISP                 = 64;   // n-disp value 260 (downscaled), 64 give caculating efficient instead of 65
int MINDISP                 = 0;

typedef struct cl_image_desc {
	cl_mem_object_type image_type;
	size_t image_width;
	size_t image_height;
	size_t image_depth;
	size_t image_array_size;
	size_t image_row_pitch;
	size_t image_slice_pitch;
	cl_uint num_mip_levels;
	cl_uint num_samples;
	cl_mem buffer;
} cl_image_desc;

cl_image_desc imgDescriptor;        // Image descriptor contains type and dimensions of the image
cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT8 };


char *read_kernel_file(const char *filename);
cl_kernel build_kernel_from_file(cl_context ctx, char const *kernel, char const *kernel_name);
void normalization(uint8_t* dispMap, uint32_t w, uint32_t h);
uint8_t* occlusion_filling(const uint8_t* dispMap, uint32_t w, uint32_t h);


int32_t main()
{
    uint8_t *OrigImageL, *OrigImageR; // Left & Right image 2940x2016
    uint8_t *dDisparity, *Disparity;

    uint32_t err;                       // Error code, 0 is OK
    uint32_t Width, Height;             // resize
    uint32_t wL, hL, wR, hR;            // original size of Left & Right image

    struct timespec totalStartTime, totalEndTime;

    cl_context ctx;
    cl_command_queue queue;
    cl_int status;

    const size_t localWorkSize[]    = {16, 16};       // Local work size
	  //const size_t localWorkSize[]    = {2, 2};       // Local work size (Odroid)
		const size_t globalWorkSize[]   = {504, 736};   // Global work size

    const size_t localWorkSize1D[]  = {localWorkSize[0]*localWorkSize[1]};      // 1-dimentional local work size
    const size_t globalWorkSize1D[] = {globalWorkSize[0]*globalWorkSize[1]};    // 1-dimentional global work size


    // ******** Load the left image into memory & check loading error ********
    err = lodepng_decode32_file(&OrigImageL, &wL, &hL, "im0.png");
    if(err) {
        printf("Error when loading the left image %u: %s\n", err, lodepng_error_text(err));
        free(OrigImageL);
        return -1;
    }
    // Load the right image into memory & check loading error
    err = lodepng_decode32_file(&OrigImageR, &wR, &hR, "im1.png");
    if(err) {
        printf("Error when loading the right image %u: %s\n", err, lodepng_error_text(err));
        free(OrigImageL);
        free(OrigImageR);
        return -1;
    }
    // Check picture size error
    if(wL!=wR || hL!=hR) {
        printf("Error, the size of left and right images not match.\n");
        free(OrigImageL);
        free(OrigImageR);
        return -1;
    }
    Width       = wL/DOWNSCALE;
    Height      = hL/DOWNSCALE;

    clock_gettime(CLOCK_MONOTONIC, &totalStartTime); // Starting time
    printf("Running openCL implement of ZNCC on images. Please wait, this will take several minutes...\n");


    // ******** Setup OpenCL environment to run the kernel ********
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

    status = clGetPlatformIDs( 1, &platform, NULL );
	printf("clGetPlatformIDs status == CL_SUCCESS - %d\n", status == CL_SUCCESS);
    int gpu = 1; // O : CPU, 1 : GPU
    status = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL);

    props[1] = (cl_context_properties)platform;
    // context
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &status );
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create context for OpenCL !\n");
        abort();
    }
    // queue
    queue = clCreateCommandQueue( ctx, device, 0, &status );
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create queue for OpenCL context !\n");
        abort();
    }


    // ******** Create buffers memory objects ********
    cl_mem clmemImageL = clCreateBuffer(ctx, CL_MEM_READ_ONLY, Width*Height, 0, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create buffer for the left image !\n");
        abort();
    }

    cl_mem clmemImageR = clCreateBuffer(ctx, CL_MEM_READ_ONLY, Width*Height, 0, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create buffer for the right image !\n");
        abort();
    }

    cl_mem clmemDispMap1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, Width*Height, 0, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create buffer for the Disparity map L vs R !\n");
        abort();
    }

    cl_mem clmemDispMap2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, Width*Height, 0, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create buffer for the Disparity map R vs L !\n");
        abort();
    }

    cl_mem clmemDispMapCrossCheck = clCreateBuffer(ctx, CL_MEM_READ_ONLY, Width*Height, 0, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create buffer for the Disparity cross checking map !\n");
        abort();
    }

    // ******** Read kernel file ********
    char *resize_kernel_file       = read_kernel_file("resize.cl");
    char *zncc_kernel_file         = read_kernel_file("zncc.cl");
    char *cross_check_kernel_file  = read_kernel_file("cross_check.cl");

    // ******* Init cl kernel from files *******
    imgDescriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
    imgDescriptor.image_width = wL;
    imgDescriptor.image_height = hL;
    imgDescriptor.image_depth = 8;
    imgDescriptor.image_row_pitch = wL * DOWNSCALE;

    dDisparity = (uint8_t*) malloc(Width*Height);
    Disparity  = (uint8_t*) malloc(Width*Height);


    cl_kernel resize_kernel     = build_kernel_from_file(ctx, resize_kernel_file, "resize");
    cl_kernel zncc_kernel       = build_kernel_from_file(ctx, zncc_kernel_file, "zncc");
    cl_kernel cross_check_kernel= build_kernel_from_file(ctx, cross_check_kernel_file, "cross_check");

    // ******** Create images memory objects ********
    cl_mem clmemOrigImageL = clCreateImage2D(ctx, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, &format, imgDescriptor.image_width, imgDescriptor.image_height, imgDescriptor.image_row_pitch, OrigImageL, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create Image for the left image !\n");
        abort();
    }

    cl_mem clmemOrigImageR = clCreateImage2D(ctx, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, &format, imgDescriptor.image_width, imgDescriptor.image_height, imgDescriptor.image_row_pitch, OrigImageR, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create Image for the right image !\n");
        abort();
    }

    // ******** Call the kernels ********
    // Resize and grayscale kernel
    status = 0;
    status  = clSetKernelArg(resize_kernel, 0, sizeof(clmemOrigImageL), &clmemOrigImageL);
    status |= clSetKernelArg(resize_kernel, 1, sizeof(clmemOrigImageR), &clmemOrigImageR);
    status |= clSetKernelArg(resize_kernel, 2, sizeof(clmemImageL), &clmemImageL);
    status |= clSetKernelArg(resize_kernel, 3, sizeof(clmemImageR), &clmemImageR);
    status |= clSetKernelArg(resize_kernel, 4, sizeof(Width), &Width);
    status |= clSetKernelArg(resize_kernel, 5, sizeof(Height), &Height);

    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to set kernel arguments for 'resize_kernel' !\n");
        abort();
    }

    status = clEnqueueNDRangeKernel(queue, resize_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);


    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to execute 'resize_kernel' on the device !\n");
        abort();
    }

    clFinish(queue);
    status = clEnqueueReadBuffer(queue, clmemImageL, CL_TRUE,  0, Width*Height, Disparity, 0, NULL, NULL);
    if(status != CL_SUCCESS){
        fprintf(stderr, "'resize_kernel': Failed to send the data to host !\n");
        abort();
    }
    status = clEnqueueReadBuffer(queue, clmemImageR, CL_TRUE,  0, Width*Height, Disparity, 0, NULL, NULL);
    if(status != CL_SUCCESS){
        fprintf(stderr, "'resize_kernel': Failed to send the data to host !\n");
        abort();
    }

    // Disparity (L vs R) ZNCC kernel
    status = 0;
    status  = clSetKernelArg(zncc_kernel, 0, sizeof(clmemImageL), &clmemImageL);
    status |= clSetKernelArg(zncc_kernel, 1, sizeof(clmemImageR), &clmemImageR);
    status |= clSetKernelArg(zncc_kernel, 2, sizeof(clmemDispMap1), &clmemDispMap1);
    status |= clSetKernelArg(zncc_kernel, 3, sizeof(Width), &Width);
    status |= clSetKernelArg(zncc_kernel, 4, sizeof(Height), &Height);
    status |= clSetKernelArg(zncc_kernel, 5, sizeof(HALFWINSIZEX), &HALFWINSIZEX);
    status |= clSetKernelArg(zncc_kernel, 6, sizeof(HALFWINSIZEY), &HALFWINSIZEY);
    status |= clSetKernelArg(zncc_kernel, 7, sizeof(WINSIZEAREA), &WINSIZEAREA);
    status |= clSetKernelArg(zncc_kernel, 8, sizeof(MINDISP), &MINDISP);
    status |= clSetKernelArg(zncc_kernel, 9, sizeof(MAXDISP), &MAXDISP);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to set kernel arguments for 'zncc_kernel', Dispmap1 !\n");
        abort();
    }

    status = clEnqueueNDRangeKernel(queue, zncc_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);

    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to execute 'zncc_kernel' on the device, Dispmap1!\n");
        abort();
    }

    // Disparity (R vs L) ZNCC kernel
    MAXDISP *= -1;
    status = 0;
    status  = clSetKernelArg(zncc_kernel, 0, sizeof(clmemImageR), &clmemImageR);
    status |= clSetKernelArg(zncc_kernel, 1, sizeof(clmemImageL), &clmemImageL);
    status |= clSetKernelArg(zncc_kernel, 2, sizeof(clmemDispMap2), &clmemDispMap2);
    status |= clSetKernelArg(zncc_kernel, 3, sizeof(Width), &Width);
    status |= clSetKernelArg(zncc_kernel, 4, sizeof(Height), &Height);
    status |= clSetKernelArg(zncc_kernel, 5, sizeof(HALFWINSIZEX), &HALFWINSIZEX);
    status |= clSetKernelArg(zncc_kernel, 6, sizeof(HALFWINSIZEY), &HALFWINSIZEY);
    status |= clSetKernelArg(zncc_kernel, 7, sizeof(WINSIZEAREA), &WINSIZEAREA);
    status |= clSetKernelArg(zncc_kernel, 8, sizeof(MAXDISP), &MAXDISP);
    status |= clSetKernelArg(zncc_kernel, 9, sizeof(MINDISP), &MINDISP);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to set kernel arguments for 'zncc_kernel' Dispmap2 !\n");
        abort();
    }

    status = clEnqueueNDRangeKernel(queue, zncc_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to execute 'zncc_kernel' on the device, Dispmap2 !\n");
        abort();
    }

    // Cross checking kernel
    status = 0;
    status  = clSetKernelArg(cross_check_kernel, 0, sizeof(clmemDispMap1), &clmemDispMap1);
    status |= clSetKernelArg(cross_check_kernel, 1, sizeof(clmemDispMap2), &clmemDispMap2);
    status |= clSetKernelArg(cross_check_kernel, 2, sizeof(clmemDispMapCrossCheck), &clmemDispMapCrossCheck);
    status |= clSetKernelArg(cross_check_kernel, 3, sizeof(THRESHOLD), &THRESHOLD);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to set kernel arguments for 'cross_check_kernel' !\n");
        abort();
    }

    status = clEnqueueNDRangeKernel(queue, cross_check_kernel, 1, NULL, (const size_t*)&globalWorkSize1D, (const size_t*)&localWorkSize1D, 0, NULL, NULL);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Failed to execute 'cross_check_kernel' on the device !\n");
        abort();
    }

    clFinish(queue);
    status = clEnqueueReadBuffer(queue, clmemDispMapCrossCheck, CL_TRUE, 0, Width*Height, dDisparity, 0, NULL, NULL);
    if(status != CL_SUCCESS){
        fprintf(stderr, "'cross_check_kernel': Failed to send the data to host !\n");
        abort();
    }


    // ******** run occlusion_filling & nomalize on host-code ********
    Disparity = occlusion_filling(dDisparity, Width, Height);
    normalization(Disparity, Width, Height);

    clock_gettime(CLOCK_MONOTONIC, &totalEndTime); // Ending time
    printf("*** Total ZNCC OpenCL executed time: %f s. ***\n", (double)(totalEndTime.tv_sec - totalStartTime.tv_sec) + (double)(totalEndTime.tv_nsec - totalStartTime.tv_nsec)/1000000000);


    // ******** Save file to working directory (setup working directory may differ from IDEs) ********
    err = lodepng_encode_file("depthmap.png", Disparity, Width, Height, LCT_GREY, 8);
    free(OrigImageR);
    free(OrigImageL);
    free(resize_kernel_file);
    free(zncc_kernel_file);
    free(cross_check_kernel_file);
    free(dDisparity);
    free(Disparity);

    clReleaseKernel(resize_kernel);
    clReleaseKernel(zncc_kernel);
    clReleaseKernel(cross_check_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    clReleaseMemObject(clmemOrigImageL);
    clReleaseMemObject(clmemOrigImageR);
    clReleaseMemObject(clmemImageL);
    clReleaseMemObject(clmemImageR);
    clReleaseMemObject(clmemDispMap1);
    clReleaseMemObject(clmemDispMap2);
    clReleaseMemObject(clmemDispMapCrossCheck);

    if(err){
        printf("Error when saving the final 'depthmap.png' %u: %s\n", err, lodepng_error_text(err));
        return -1;
    }
    return 0;
}

/******************************************************************************
 *  Function that use to read kernel file
 */
char *read_kernel_file(const char *filename)
{
    FILE *f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *res = (char *) malloc(size+1);
    if(!res) { // check error
        perror("Fail to read file, can not allocation memory !");
        abort();
    }

    // read file from stream
    if(fread(res, 1, size, f) < size) { // check error
        perror("Fail to read file, fread abort !");
        abort();
    }

    fclose(f);
    res[size] = '\0';
    return res;
}

/******************************************************************************
 *  Function that use to build kernel from file
 */
cl_kernel build_kernel_from_file(cl_context ctx, char const *kernel, char const *kernel_name)
{
    cl_int status;

    printf("Building kernel '%s'\n", kernel_name);
    // get kernel size
    size_t sizes[] = { strlen(kernel) };

    // create program
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel, sizes, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create program, in kernel file: '%s' !\n", kernel_name);
        abort();
    }

    // build program
    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to build program, in kernel file: '%s' !\n", kernel_name);
        abort();
    }

    // create our kernel from program
    cl_kernel res = clCreateKernel(program, kernel_name, &status);
    if(status != CL_SUCCESS){
        fprintf(stderr, "Fail to create kernel, in kernel file: '%s' !\n", kernel_name);
        abort();
    }
    clReleaseProgram(program);
    printf("Succefully builed kernel '%s'\n", kernel_name);
    return res;
}

/******************************************************************************
 *  Replace each pixel with zero value with the nearest non-zero pixel value
 */
uint8_t* occlusion_filling(const uint8_t* dispMap, uint32_t w, uint32_t h) {
    int32_t i, j, ii, jj, k;
    bool flag; // flag for nearest non-zero pixel value

    uint8_t* result = (uint8_t*) malloc(w*h);

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // If the value of the pixel is zero, perform the occlusion filling by nearest non-zero pixel value
            result[i*w+j] = dispMap[i*w+j];
            if(dispMap[i*w+j] == 0) {
                // Search of non-zero pixel in the neighborhood i,j, neighborhoodsize++
                flag = true;
                k = 0;
                while(flag) {
                    k++;
                    jj = -k;
                    for (ii = -k; ii <= k && flag; ii++) {
                        if (0<=i+ii && i+ii<h && 0<=j+jj && j+jj<w && dispMap[(i+ii)*w+(j+jj)]!=0) {
                            result[i*w+j] = dispMap[(i+ii)*w+(j+jj)];
                            flag = false;
                            break;
                        }
                    }
                    jj = k;
                    for (ii = -k; ii <= k && flag; ii++) {
                        if (0<=i+ii && i+ii<h && 0<=j+jj && j+jj<w && dispMap[(i+ii)*w+(j+jj)]!=0) {
                            result[i*w+j] = dispMap[(i+ii)*w+(j+jj)];
                            flag = false;
                            break;
                        }
                    }
                    ii = -k;
                    for (jj = -k+1; jj <= k-1 && flag; jj++) {
                        if (0<=i+ii && i+ii<h && 0<=j+jj && j+jj<w && dispMap[(i+ii)*w+(j+jj)]!=0) {
                            result[i*w+j] = dispMap[(i+ii)*w+(j+jj)];
                            flag = false;
                            break;
                        }
                    }
                    ii = k;
                    for (jj = -k+1; jj <= k && flag; jj++) {
                        if (0<=i+ii && i+ii<h && 0<=j+jj && j+jj<w && dispMap[(i+ii)*w+(j+jj)]!=0) {
                            result[i*w+j] = dispMap[(i+ii)*w+(j+jj)];
                            flag = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    return result;
}

/******************************************************************************
 *  Normalize the final disparity map
 */
void normalization(uint8_t* dispMap, uint32_t w, uint32_t h) {
    uint8_t maxValue = 0, minValue = UCHAR_MAX;
    uint32_t i;
    for (i = 0; i < w*h; i++) {
        if(dispMap[i]>maxValue) {maxValue=dispMap[i];}
        if(dispMap[i]<minValue) {minValue=dispMap[i];}
    }
    // Nomarlize to grey scale 0..255(UCHAR_MAX)
    maxValue -= minValue;
    for (i = 0; i < w*h; i++) {
        dispMap[i] = (UCHAR_MAX*(dispMap[i] - minValue)/maxValue);
    }
}
