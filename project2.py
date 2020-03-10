import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob         # for reading all calibration images 

# read in the calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# calibrate the camera
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image space

def find_camera_params():
    # since (10,7) image they will have obj points from (0,0,0,) to (9,6,0) 
    # first prepare obj points
    objp = np.zeros((ny*nx,3), np.float32) # Return a new array of given shape and type, filled with zeros
    # Fill the grid with values to (9,6), make a transpose of it and reshape it back to 2 columns, x and y
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # draw the chesboard corners for one image
    # f, axes= plt.subplots(1,2,figsize=(30,30))
    
    for index, fname in enumerate(images):
        # Read in an image
        img = cv2.imread(fname)

        # find chessboard corners after converting to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        #if corners found, add to imgpoints and objpoints
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # draw the found corners. img is changed, no return required
            cv2.drawChessboardCorners(img, (9,6), corners, ret)

            #draw one image with the corners
            if(index==1 ):
                print("printing")
                plt.imshow(img)
                # axes[0].imshow(draw)
                # axes[0].set_title('chessboard corners', fontsize=20)
                # axes[1].imshow(draw)
                # axes[1].set_title('chessboard corners', fontsize=20)
                plt.show()

find_camera_params()

    
# def get_pers_trans_matrix(img,corners):
#     #get perspective transform matrix
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_size = (gray.shape[1], gray.shape[0])
#     offset = 100
#     src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#     dst = np.float32([[offset,offset], [img_size[0]-offset,offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
#     M = cv2.getPerspectiveTransform(src,dst)
    
#     return M
    

# #function that takes an image, object points, and image points
# # performs the camera calibration, image distortion correction and 
# # returns the undistorted image
# def cal_undistort(img, objpoints, imgpoints):
#     # Use cv2.calibrateCamera() and cv2.undistort()
    
    
    
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     return undist
    
# # Apply color transforms and gradients to the images
# def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
#     img = np.copy(img)
#     # Convert to HLS color space and separate the V channel
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     l_channel = hls[:,:,1]
#     s_channel = hls[:,:,2]
#     # Sobel x
#     sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
#     # Threshold x gradient
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
#     # Threshold color channel
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
#     # Stack each channel
#     color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255  # return this if you want to check the thresholds
    
#     # Combine the two binary thresholds
#     combined_binary = np.zeros_like(sxbinary)
#     combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
#     return combined_binary

# def get_vertices(image):
#     ysize = image.shape[0]
#     xsize = image.shape[1]
#     bottom_left_x = 0.1*xsize
#     bottom_left_y = 0.9*ysize
#     top_left_x = 0.5*xsize      # comes to 384
#     top_left_y = 0.6*ysize      # comes to 324
#     bottom_right_x = 0.9*xsize
#     bottom_right_y = 0.9*ysize
#     top_right_x = 0.6*xsize
#     top_right_y = 0.6*ysize
#     vertices = np.array([[[bottom_left_x, bottom_left_y], [top_left_x,top_left_y],[top_right_x,top_right_y], [bottom_right_x,bottom_right_y]]], dtype=np.int32)
#     return vertices
    
# def region_of_interest(img, vertices):
#     """
#     Applies an image mask.
    
#     Only keeps the region of the image defined by the polygon
#     formed from `vertices`. The rest of the image is set to black.
#     `vertices` should be a numpy array of integer points.
#     """
#     #defining a blank mask to start with
#     mask = np.zeros_like(img)   
    
#     #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255
        
#     #filling pixels inside the polygon defined by "vertices" with the fill color    
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
    
#     #returning the image only where mask pixels are nonzero
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

# def canny(img, low_threshold, high_threshold):
#     """Applies the Canny transform"""
#     return cv2.Canny(img, low_threshold, high_threshold)    
    
# def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
#     """
#     This function draws `lines` with `color` and `thickness`.    
#     Lines are drawn on the image inplace (mutates the image).
#     If you want to make the lines semi-transparent, think about combining
#     this function with the weighted_img() function below
#     """
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
#     """
#     `img` should be the output of a Canny transform.
        
#     Returns an image with hough lines drawn.
#     """
#     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     draw_lines(line_img, lines)
#     return line_img 
    

# def corners_unwarp(img, nx, ny, mtx, dist):
#     # Use the OpenCV undistort() function to remove distortion
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     img_size = (undist.shape[1], undist.shape[0])

#     # For source points I'm grabbing the outer four detected corners
#     src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#     # For destination points, I'm arbitrarily choosing some points to be
#     # a nice fit for displaying our warped result 
#     # again, not exact, but close enough for our purposes
#     dst = np.float32([[offset, offset], [img_size[0]-offset, offset],[img_size[0]-offset, img_size[1]-offset],[offset, img_size[1]-offset]])
#     # Given src and dst points, calculate the perspective transform matrix
#     M = cv2.getPerspectiveTransform(src, dst)
#     # Warp the image using OpenCV warpPerspective()
#     warped = cv2.warpPerspective(undist, M, img_size)

#     # Return the resulting image and matrix
#     return warped, M

# img = cv2.imread('./camera_cal/calibration3.jpg')
# img_size = (img.shape[1], img.shape[0])
# find_camera_params()
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
# warped,perspective_M  = corners_unwarp(img, nx, ny, mtx, dist)

# img = cv2.imread('./test_images/straight_lines1.jpg')
# img_size = (img.shape[1], img.shape[0])
# undist = cv2.undistort(img, mtx, dist, None, mtx)
# warped2 = cv2.warpPerspective(undist, perspective_M, img_size,flags=cv2.INTER_LINEAR)

# """
# img = cv2.imread('./test_images/straight_lines1.jpg')
# img_size = (img.shape[1], img.shape[0])
# undist = cv2.undistort(img, mtx, dist, None, mtx)
# col_grad = color_and_gradient(undist)
# warped = cv2.warpPerspective(col_grad, perspective_M, img.shape[::-1],flags=cv2.INTER_LINEAR)
# """
# """
# # get the vertices for the ROI
# vertices = get_vertices(col_grad)
# print(vertices)

# masked_image = region_of_interest(col_grad,vertices)
# """

# result = warped2
# # use (arr, cmap='gray') option when you want to show a grey scale image 
# plt.imshow(result, cmap='gray')
# plt.show()
