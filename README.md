OpenCL-implementation of the Zero-mean Normalized Cross Correlation(ZNCC)


	+ Decode two 32-bit RGBA images                                (running on host-code)
	+ Resize these images by 1/16 (from 2940x2016 to 735x504)
	+ Transform these images to greyscale images
	+ Implement ZNCC on these image with changeable window size, output of ZNCC is disparity map.
	+ Cross check two output disparity maps
	+ Occlusion filling one output disparity map from cross check  (running on host-code)
	+ Normalize the disparity map to 0..255                        (running on host-code)
	   (case after downscaled, MAXDISP is 64)
	+ Output the result image to "depthmap.png"                    (running on host-code)



NOTES :
	+ Make use of the lodepng lib: http://lodev.org/lodepng/

AUTHOR :    Lam Huynh


MAJOR CHANGES : (checkout git for details)


NO		VERSION		DATE			DETAIL
01		0.1.1		10Mar17			Respawned host-code from C-implementation
02		0.1.2		17Mar17			resize.cl corrected
03		0.1.3		21Mar17			zncc.cl corrected
04		0.1.4		26Mar17			cross_check.cl corrected
05		0.1.5		31Mar17			occlusion_filling.cl worked
06		0.1.5		05Apr17			normalize.cl worked
07		0.2.1		10Apr17			remove occlusion_filling.cl & normalize.cl (got stuck in optimize)
08		0.2.2		20Apr17			change host-code
08		0.2.3		05May17			Final host-code
