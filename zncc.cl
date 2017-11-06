__kernel void zncc(__global uchar *leftImg, __global  uchar *rightImg, __global uchar *dispMap, int w, int h,  int halfwinsizex, int halfwinsizey, int winsizearea, int mind, int maxd) {
    
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    
    int ii, jj, d, best_d; //declare idx, d is disparity value
    float avgLeft, avgRight, leftWinValue, rightWinValue, leftStdDeviation, rightStdDeviation;
    float currZNCC, bestZNCC; // current and best ZNCC value
    
    // Searching for d with best ZNCC score for the each pixels
    best_d = maxd;
    bestZNCC = -1;
    for (d = mind; d <= maxd; d++) {
        // Calculating the window average
        avgLeft = avgRight = 0;
        for (ii = -halfwinsizey; ii < halfwinsizey; ii++) {
            for (jj = -halfwinsizex; jj < halfwinsizex; jj++) {
                if (0<=i+ii && i+ii<h && 0<=j+jj && j+jj<w && 0<=j+jj-d && j+jj-d<w) {
                    // Sum all pixels in window size
                    avgLeft  += leftImg [(i+ii)*w + (j+jj)];
                    avgRight += rightImg[(i+ii)*w + (j+jj-d)];
                }
            }
        }
        avgLeft  /= winsizearea;
        avgRight /= winsizearea;
        leftStdDeviation = rightStdDeviation = currZNCC = 0;
        
        // Calculate using the ZNCC formula
        for (ii = -halfwinsizey; ii < halfwinsizey; ii++) {
            for (jj = -halfwinsizex; jj < halfwinsizex; jj++) {
                if (0<=i+ii && i+ii<h && 0<=j+jj && j+jj<w && 0<=j+jj-d && j+jj-d<w) {
                    leftWinValue       = leftImg[(i+ii)*w + (j+jj)] - avgLeft;
                    rightWinValue      = rightImg[(i+ii)*w + (j+jj-d)] - avgRight;
                    currZNCC          += leftWinValue*rightWinValue;
                    leftStdDeviation  += leftWinValue*leftWinValue;
                    rightStdDeviation += rightWinValue*rightWinValue;
                }
            }
        }
        // Calculate current ZNCC value
        currZNCC /= native_sqrt(leftStdDeviation)*native_sqrt(rightStdDeviation);
        // Winner-takes-it-all-approach, get d with the best ZNCC value
        if (currZNCC > bestZNCC) {
            bestZNCC = currZNCC;
            best_d = d;
        }
    }
    dispMap[i*w+j] = (uint)abs(best_d);
}
