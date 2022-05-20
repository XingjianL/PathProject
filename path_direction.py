import cv2
import numpy as np
from scipy import ndimage
import os

PATH_THRESHOLD = 130     # threshold value for gray to binary
FORWARD_DEFAULT = [0,-1] # image up is forward
# input - image with partial path
# steps
#   1. convert to binary/grayscale image
#   2. sum the edge pixel values
#   3. if one edge is not 0 (or some threshold), move towards that edge
#   4. else process input as entire path

# input - image with entire path    (should be done except realistic situation of step 1)
# steps
#   1. convert to binary image (kmeans/grayscale -> threshold -> erosion if needed)
#   2. Extract two sections of path -> coordinate PCA, second component
#   3. Extract direction of each section -> coordinate PCA, first component

### 
#   Functions
###
# input: binary image
# output: mean: center of the coordinates, 
#         pca_vector: [[PC2_x, PC1_x], [PC2_y, PC1_y]]
def Path_PCA(image):                       # definition method
    pca_vector = []
    #image = cv2.resize(image,IMAGE_SIZE)
    coords_data = np.array(cv2.findNonZero(image)).T.reshape((2,-1))            # 2 x n matrix of coords [[x1,x2,...],[y1,y2,...]]
    mean = np.mean(coords_data,axis=1,keepdims=True)                         # center of coords
    cov_mat = np.cov(coords_data - mean, ddof = 1)              # find covariance
    pca_val, pca_vector = np.linalg.eig(cov_mat)                # find eigen vectors (also PCA first and second component)
    return mean, pca_vector, pca_val

# changes the value above the line in an image
# input: image_mask, initial_coord[x,y], slope[x,y], value= 0,1 (for binary masking)
def set_mask(image_mask, initial_coord, slope, value):
    # line_y = mx+b
    # b = line_y - mx
    m = slope[1] / slope[0]
    b = initial_coord[1] - m * initial_coord[0]
    for x in range(image_mask.shape[1]):
        line_y = int(round(m*x + b))
        if line_y > image_mask.shape[0]: line_y = image_mask.shape[0]
        if line_y < 0: line_y = 0
        image_mask[0:line_y, x] = value # change value of under the line (top of the image)
    return image_mask

# just for finding the place to draw the circle
def compute_location(pca_cent, pca_dir, scale = 10):
    return (int(pca_cent[0] + scale * pca_dir[0]),
            int(pca_cent[1] + scale * pca_dir[1]))

# compute angle between two vectors
# arccos((unit_a dot unit_b))
def compute_angle(v_1, v_2):
    unit_v_1 = v_1 / np.linalg.norm(v_1)
    unit_v_2 = v_2 / np.linalg.norm(v_2)
    return np.arccos(np.dot(unit_v_1,unit_v_2))


path_dir = '/home/lixin/Projects/GitHubProjects/PathProject/Data/Screenshot 2022-05-20 185437.png'
if __name__ == '__main__':

    frame = cv2.imread(path_dir)
    
    #frame = cv2.resize(frame, IMAGE_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # simple threshold
    _, thres = cv2.threshold(gray, PATH_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('thres',thres)
    input_shape = thres.shape
    print(input_shape)
    # compute Principle Components
    # to find line between two paths
    center1, pca_vector_1, pca_val = Path_PCA(thres)

    # display the overall PC
    hori_cent, vert_cent = int(center1[0][0]), int(center1[1][0])
    pca1_loc = compute_location(center1[:,0],pca_vector_1[:,1])
    pca2_loc = compute_location(center1[:,0],pca_vector_1[:,0])
    cv2.circle(gray,(hori_cent, vert_cent),radius=3,color=(0))
    cv2.circle(gray,pca1_loc,radius=3,color=(255))
    cv2.circle(gray,pca2_loc,radius=3,color=(255))
    cv2.imshow('center', gray)
    #print('initial coord: ', center1)
    #print('slope ', pca_vector_1)
    #print('pca val ', pca_val)
    slice_dir = np.argmin(pca_val)  # slice by the minimum variance
    # Create the masks to separate two paths
        # may need to use first components if the 2 path form a right angle  (PC1, pca_vector_1[:,1] vs PC2, [:,0])
        # else using the second is fine
    mask_one = np.ones(input_shape, dtype="uint8")                                   # generate mask
    mask_one = set_mask(mask_one, center1[:,0], pca_vector_1[:,slice_dir], 0)       # set above 0
    mask_two = np.zeros(input_shape, dtype="uint8")                                  # generate mask
    mask_two = set_mask(mask_two, center1[:,0], pca_vector_1[:,slice_dir], 1)       # set above 1
    # generate the two path segments
    bottom_path = cv2.bitwise_and(thres,thres,mask=mask_one)
    top_path = cv2.bitwise_and(thres,thres,mask=mask_two)
    
    cv2.imshow('mask1_path',bottom_path)
    cv2.imshow('mask2_path',top_path)

    # Compute Principle Components for both path segments (center point(mean), direction vector(eigvec), variance vector(eigval))
    path_center1, path_direction1, pca_val1 = Path_PCA(bottom_path)
    path_center2, path_direction2, pca_val2 = Path_PCA(top_path)
    # select highest variance for each(eigenvalue)
    bot_dir = path_direction1[:,np.argmax(pca_val1)]
    top_dir = path_direction2[:,np.argmax(pca_val2)]
    # center of two segments (start location)
    bot_hori_cent, bot_vert_cent = int(path_center1[:,0][0]), int(path_center1[:,0][1])
    top_hori_cent, top_vert_cent = int(path_center2[:,0][0]), int(path_center2[:,0][1])
    # compute end location of two segment directions
    bot_pca_1 = compute_location(path_center1.T[0],bot_dir, scale = 20)
    top_pca_1 = compute_location(path_center2.T[0],top_dir, scale = 20)

    # find angle of bottom direction and top direction with respect to up
    bot_angle = compute_angle(FORWARD_DEFAULT, bot_dir)
    angle = compute_angle(FORWARD_DEFAULT, top_dir)
    print("+x is right, -y is up")
    print(f"bot_dir[x,y]: {bot_dir}, top_dir[x,y]: {top_dir}, bot_angle(rad): {bot_angle}, top_angle: {angle}")
    
    
    cv2.putText(frame,'bot',bot_pca_1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(255,255,255))
    cv2.putText(frame,'top',top_pca_1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(255,255,255))
    cv2.arrowedLine(frame,(bot_hori_cent, bot_vert_cent),bot_pca_1,
                    color=(0,0,0),thickness=2,tipLength=0.5)
    cv2.arrowedLine(frame,(top_hori_cent, top_vert_cent),top_pca_1,
                    color=(255,255,255),thickness=2,tipLength=0.5)
    cv2.imshow('final', frame)

    rotated_bot_up = ndimage.rotate(frame,bot_angle*180/np.pi) # rotate the image so the top is vertical
    rotated_top_up = ndimage.rotate(frame,angle*180/np.pi) # rotate the image so the bot is vertical
    cv2.imshow('after_bot', rotated_bot_up)
    cv2.imshow('after_top', rotated_top_up)
    cv2.imwrite(os.getcwd()+'/Results/labeled_result.png', frame)
    cv2.imwrite(os.getcwd()+'/Results/rotated_result_bot.png',rotated_bot_up)
    cv2.imwrite(os.getcwd()+'/Results/rotated_result_top.png',rotated_top_up)
    cv2.waitKey()
    
    input()
