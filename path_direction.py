
import cv2
import numpy as np
from scipy import ndimage
import os

class ImagePrep:
    KMEANSFILTER = [3,  # num of clusters
                4,  # num of iterations
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), # criteria
                cv2.KMEANS_PP_CENTERS]  # flag

    def __init__(self, slice_size = 10, kmeans_filter = KMEANSFILTER):
        self.slice_size = slice_size
        self.k, self.iter_num, self.criteria, self.flag = kmeans_filter
    def slice(self, image):
        arr_size = tuple(int(element / self.slice_size) for element in image.shape)
        col_array = np.array_split(image, arr_size[0], axis=0)
        img_array = []
        for col in col_array:
            img_array.append(np.array_split(col,arr_size[1],axis=1))
        return img_array
    def combineRow(self,imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=1)
        return combined_img
    def combineCol(self,imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=0)
        return combined_img
    def reduce_image_color(self, image):
        img_kmean = image.reshape(-1,3)
        img_kmean = np.float32(img_kmean)
        ret,label,center = cv2.kmeans(img_kmean,self.k,None,self.criteria,self.iter_num,self.flag)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        return res2, center

SCALING_FACTOR = 0.5

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
        # compute y
        line_y = int(round(m*x + b))
        # bound y within image height
        if line_y > image_mask.shape[0]: line_y = image_mask.shape[0]
        if line_y <= 0: line_y = 1
        # change value of under the line (top of the image)
        image_mask[0:line_y, x] = value 
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



if __name__ == '__main__':
    path_dir = '/home/lixin/Projects/GitHubProjects/PathProject/Data/Screen Shot 2022-05-28 at 3.50.20 PM.png'

    test_prep = ImagePrep(slice_size = 50)
    
    ####
    #   Filter Image to Binary Image
    ####
    frame = cv2.imread(path_dir)

    width = int(frame.shape[1] * SCALING_FACTOR)
    height = int(frame.shape[0] * SCALING_FACTOR)
    frame = cv2.resize(frame, (width, height))

    test_slice_imgs = test_prep.slice(frame)
    test_kmeans = test_slice_imgs.copy()
    comb_row = [i for i in range(len(test_slice_imgs))]
    for i,row in enumerate(test_slice_imgs):
        for j,block in enumerate(row):
            test_kmeans[i][j], _ = test_prep.reduce_image_color(block)
        comb_row[i] = (test_prep.combineRow(test_kmeans[i]))
    combined_filter = test_prep.combineCol(comb_row)
    filter_final, colors = test_prep.reduce_image_color(combined_filter)
    cv2.imshow('testslice',filter_final)

    ####
    #   Find Path Directions
    ####
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(filter_final, cv2.COLOR_BGR2GRAY) 
    cv2.imshow('g',gray)    
    gray_colors, gray_counts = np.unique(gray.flatten(),return_counts=True)                 # find the colors and counts of each color
    path_color = gray_colors[np.argsort(gray_counts)[0]]                                    # find the least common color
    print(gray_colors,gray_counts,path_color)
    # simple threshold, use the least frequent color (which should not be the background)
    thres = np.uint8(np.where(gray == path_color, 255, 0))
    #_, thres = cv2.threshold(gray, PATH_THRESHOLD_HIGH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

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

    slice_dir = np.argmin(pca_val)  # slice by the minimum variance
    # Create the masks to separate two paths
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

    rotated_bot_up = ndimage.rotate(frame,bot_angle*180/np.pi) # rotate the image so the bot is vertical
    rotated_top_up = ndimage.rotate(frame,angle*180/np.pi) # rotate the image so the top is vertical
    cv2.imshow('after_bot', rotated_bot_up)
    cv2.imshow('after_top', rotated_top_up)
    print(os.getcwd())
    cv2.imwrite(os.getcwd()+'/Results/labeled_result.png', frame)
    cv2.imwrite(os.getcwd()+'/Results/rotated_result_bot.png',rotated_bot_up)
    cv2.imwrite(os.getcwd()+'/Results/rotated_result_top.png',rotated_top_up)
    cv2.waitKey()
    
    input()
