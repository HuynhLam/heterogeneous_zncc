__constant sampler_t tmp = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

__kernel void resize(__read_only image2d_t origImgL, __read_only image2d_t origImgR, __global uchar *resImgL, __global  uchar *resImgR, int scale_w, int scale_h) {
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
    // Red index[i][j]
    int2 redIdx = { (4*j - 1*(j > 0)), (4*i - 1*(i > 0)) };
    // Grayscaling
    uint4 pixelLeftImage  = read_imageui(origImgL, tmp, redIdx);
    uint4 pixelRightImage = read_imageui(origImgR, tmp, redIdx);
    
    resImgL[i*scale_w+j] = 0.2126*pixelLeftImage.x  + 0.7152*pixelLeftImage.y  + 0.0722*pixelLeftImage.z;
    resImgR[i*scale_w+j] = 0.2126*pixelRightImage.x + 0.7152*pixelRightImage.y + 0.0722*pixelRightImage.z;
}
