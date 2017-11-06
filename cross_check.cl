__kernel void cross_check(__global uchar* dispMap1, __global uchar* dispMap2, __global uchar* res, uint threshold) {
    const int i = get_global_id(0);
    // Checking abs(diff(dispMap1 & dispMap2)) at each pixels
    // Dispose all the diff exceed threshold values at each pixels
    if (abs((int)dispMap1[i] - dispMap2[ i-dispMap1[i] ]) > threshold)
        res[i] = 0;
    else
        res[i] = dispMap1[i];
}
