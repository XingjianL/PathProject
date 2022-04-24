import cv2
import numpy as np

IMAGE_SIZE = (300,300)    # (width(x), height(y))
PATH_THRESHOLD = 130     # threshold value for gray to binary
# input - image with partial path
# steps
#   1. convert to binary/grayscale image
#   2. sum the edge pixel values
#   3. if one edge is not 0 (or some threshold), move towards that edge
#   4. else process input as entire path

# input - image with entire path    (mostly done, need to take care of the edge case of two paths forming right angle)
# steps
#   1. convert to binary image (kmeans/grayscale -> threshold -> erosion if needed)
#   2. Extract two sections of path -> coordinate PCA, second component
#   3. Extract direction of each section -> coordinate PCA, first component

# solving above edge case (right angle paths)
#   1. check two path direction after first check (if same -> use first component, else correct vectors)
#   2. second check (if 90 degrees -> correct vectors, else, first check was correct vectors)


### 
#   Functions
###
# input: binary image
# output: mean: center of the coordinates, 
#         pca_vector: [[PC2_x, PC1_x], [PC2_y, PC1_y]]
def Path_PCA(image):                       # definition method
    pca_vector = []
    image = cv2.resize(image,IMAGE_SIZE)
    coords_data = np.array(cv2.findNonZero(image)).T.reshape((2,-1))            # 2 x n matrix of coords [[x1,x2,...],[y1,y2,...]]
    mean = np.mean(coords_data,axis=1,keepdims=True)                         # center of coords
    print(mean)
    print(np.mean(coords_data - mean, axis=1,keepdims=True))
    cov_mat = np.cov(coords_data - mean, ddof = 1)              # find covariance
    print(coords_data.shape)
    pca_val, pca_vector = np.linalg.eig(cov_mat)                # find eigen vectors (also PCA first and second component)
    return mean, pca_vector

# changes the value above the line in an image
# input: image_mask, initial_coord[x,y], slope[x,y], value= 0,1 (for binary masking)
def set_mask(image_mask, initial_coord, slope, value):
    # line_y = mx+b
    # b = line_y - mx
    m = slope[1] / slope[0]
    b = initial_coord[1] - m * initial_coord[0]
    for x in range(IMAGE_SIZE[0]):
        line_y = int(round(m*x + b))
        image_mask[0:line_y, x] = value # change value of under the line (top of the image)
    return image_mask

path_dir = '/home/xing/TesterCodes/OpenCV/PathProject/testpath2.png'
if __name__ == '__main__':

    frame = cv2.imread(path_dir)
    frame = cv2.resize(frame, IMAGE_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # simple threshold
    _, thres = cv2.threshold(gray, PATH_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('thres',thres)

    # compute Principle Components
    # to find line between two paths
    center1, pca_vector_1 = Path_PCA(thres)

    # display the overall PC
    hori_cent, vert_cent = int(center1[0][0]), int(center1[1][0])
    pca1_hori, pca1_vert  = int(center1[0][0] + 10 * pca_vector_1[:,1][0]),int(center1[1][0] + 10 * pca_vector_1[:,1][1])
    pca2_hori, pca2_vert  = int(center1[0][0] + 10 * pca_vector_1[:,0][0]),int(center1[1][0] + 10 * pca_vector_1[:,0][1])
    cv2.circle(gray,(hori_cent, vert_cent),radius=3,color=(0))
    cv2.circle(gray,(pca1_hori, pca1_vert),radius=3,color=(255))
    cv2.circle(gray,(pca2_hori, pca2_vert),radius=3,color=(255))
    cv2.imshow('center', gray)
    #print((pca2_hori,pca2_vert))
    #print('initial coord: ', center1[:,0])
    #print('slope', pca_vector_1[:,0])

    # Create the masks to separate two paths
        # may need to use first components if the 2 path form a right angle  (PC1, pca_vector_1[:,1] vs PC2, [:,0])
        # else using the second is fine
    mask_one = np.ones(IMAGE_SIZE, dtype="uint8")                          # generate mask
    mask_one = set_mask(mask_one, center1[:,0], pca_vector_1[:,0], 0)       # set above 0
    mask_two = np.zeros(IMAGE_SIZE, dtype="uint8")                          # generate mask
    mask_two = set_mask(mask_two, center1[:,0], pca_vector_1[:,0], 1)       # set above 1
    # generate the two path segments
    bottom_path = cv2.bitwise_and(thres,thres,mask=mask_one)
    top_path = cv2.bitwise_and(thres,thres,mask=mask_two)
    
    cv2.imshow('mask1_path',bottom_path)
    cv2.imshow('mask2_path',top_path)

    # Compute Principle Components for both path segments
    path_center1, path_direction1 = Path_PCA(bottom_path)
    path_center2, path_direction2 = Path_PCA(top_path)
    
    bot_hori_cent, bot_vert_cent = int(path_center1[:,0][0]), int(path_center1[:,0][1])
    top_hori_cent, top_vert_cent = int(path_center2[:,0][0]), int(path_center2[:,0][1])

    bot_path_hori, bot_path_vert  = int(path_center1[:,0][0] + 20 * path_direction1[:,1][0]),\
                                    int(path_center1[:,0][1] + 20 * path_direction1[:,1][1])
    top_path_hori, top_path_vert  = int(path_center2[:,0][0] + 20 * path_direction2[:,1][0]),\
                                    int(path_center2[:,0][1] + 20 * path_direction2[:,1][1])
    print(path_direction1)
    print(path_direction2)
    cv2.circle(frame,(bot_hori_cent, bot_vert_cent),radius=3,color=(0,0,0))
    cv2.circle(frame,(top_hori_cent, top_vert_cent),radius=3,color=(0,0,0))
    cv2.circle(frame,(bot_path_hori, bot_path_vert),radius=4,color=(255,255,0))
    cv2.circle(frame,(top_path_hori, top_path_vert),radius=4,color=(255,255,255))
    cv2.imshow('final', frame)
    cv2.waitKey()
    
    input()