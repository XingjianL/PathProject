###########
#   This file removes the imshow, storing files, and some debugging tools before translating into ROS project
#   there are three classes, ImagePrep, Path_Properties, and path_direction
# 
########### 
import cv2
import numpy as np

class ImagePrep:
    KMEANSFILTER = [3,  # num of clusters
                4,  # num of iterations
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), # criteria
                cv2.KMEANS_PP_CENTERS]  # flag

    def __init__(self, slice_size = 10, kmeans_filter = KMEANSFILTER):
        self.slice_size = slice_size
        self.k, self.iter_num, self.criteria, self.flag = kmeans_filter

    def slice(self, image):
        arr_size = tuple(int(element / self.slice_size) for element in image.shape) # number of slices in each direction
        col_array = np.array_split(image, arr_size[0], axis=0)                      # slice into columns
        img_array = []
        for col in col_array:
            img_array.append(np.array_split(col,arr_size[1],axis=1))                # slice into blocks
        return img_array

    def combineRow(self, imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=1)
        return combined_img

    def combineCol(self, imgs):
        combined_img = imgs[0]
        for img in imgs[1:]:
            combined_img = np.concatenate((combined_img,img),axis=0)
        return combined_img

    # return res2 as image with ncluster or self.k amount of colors
    def reduce_image_color(self, image, ncluster = None):
        img_kmean = image.reshape(-1,3)
        img_kmean = np.float32(img_kmean)
        if ncluster is not None:
            ret,label,center = cv2.kmeans(img_kmean,ncluster,None,self.criteria,self.iter_num,self.flag)
        else:
            ret,label,center = cv2.kmeans(img_kmean,self.k,None,self.criteria,self.iter_num,self.flag)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        return res2, center

# class for some valuable properties (color, direction, location)
# like struct
class Path_Properties:
    # expected difference between comparisons [path_color(color), background_color(color), path_theta(radians), size(% of image), location_x(pixels), location_y(pixels)]
    PATH_PROPERTY_THRESHOLDS = np.array([10, 20, np.pi/6, 0.04, 10, 10]) 
    # respective tags for the properties
    PROPERTY_TAGS = ["path_color", "background_color", "bot2top_theta", "path_size", "x", "y"]
    def __init__(self, path_properties = None):
        self.confidence = 0.5

        if path_properties is not None:
            self.properties = path_properties
            return
            
        self.path_color = None
        self.background_color = None
        self.path_theta = None
        self.size = None
        self.x = None
        self.y = None
        self.properties = np.array([self.path_color, self.background_color, self.path_theta, self.size, self.x, self.y])

    # pass in a numpy array of properties
    def compareAndUpdate(self, path_properties):
        # no property for this path
        if (self.properties == None).any():
            self.properties = path_properties
            return True
        # within threshold (update property and increase confidence)
        if self.withinThreshold(path_properties, self.properties, self.PATH_PROPERTY_THRESHOLDS):
            self.properties = path_properties
            if self.confidence < 1: 
                self.confidence += 0.01
            return True
        # outside of threshold (no update and decrease confidence)
        if self.confidence > 0: 
            self.confidence -= 0.01
        return False

    def withinThreshold(self, val1, val2, thres):
        print(abs(val1 - val2))
        return (abs(val1 - val2) < thres).all() # return True only when all properties are under the threshold


# class that actually finds the direction of the path by PCA

class path_direction:

    SCALING_FACTOR = 0.2

    NOISE_PROPORTION = 0.05 # threshold % of the image as path (less means it is noise)
    PATH_THRESHOLD = 130     # threshold value for gray to binary
    FORWARD_DEFAULT = [0,-1] # image up is forward

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

    # compute angle between two vectors
    # arccos((unit_a dot unit_b))
    def compute_angle(v_1, v_2):
        unit_v_1 = v_1 / np.linalg.norm(v_1)
        unit_v_2 = v_2 / np.linalg.norm(v_2)
        return np.arccos(np.dot(unit_v_1,unit_v_2))

    def compute_slope(p_1, p_2):
        return (p_2[0] - p_1[0]), (p_2[1] - p_1[1])


    ####
    #   for init and frame_callback
    ####
    if __name__ == '__main__':
        # init stuff
        test_prep = ImagePrep(slice_size = 50)

        # path number 1
        path_object = Path_Properties() 
        ####
        #   Filter Image to Binary Image
        #   
        #   Start of frame 
        ####
        frame = cv2.imread()

        width = int(frame.shape[1] * SCALING_FACTOR)
        height = int(frame.shape[0] * SCALING_FACTOR)
        frame = cv2.resize(frame, (width, height))

        test_slice_imgs = test_prep.slice(frame)
        test_kmeans = test_slice_imgs.copy()
        comb_row = [i for i in range(len(test_slice_imgs))]
        for i,row in enumerate(test_slice_imgs):
            for j,block in enumerate(row):
                test_kmeans[i][j], _ = test_prep.reduce_image_color(block, print_time=False)
            comb_row[i] = (test_prep.combineRow(test_kmeans[i]))
        combined_filter = test_prep.combineCol(comb_row)

        combined_filter[:,:,1:3] = 0    # only blue channel is relevant (clear GR in BGR)

        filter_final, colors = test_prep.reduce_image_color(combined_filter,2)  # reduce to 2 colors (background and path)

        ####
        #   Find Path Directions
        ####
        gray = cv2.cvtColor(filter_final, cv2.COLOR_BGR2GRAY) 

        # adaptively find the path color
        gray_colors, gray_counts = np.unique(gray.flatten(),return_counts=True)                 # find the colors and counts of each color
        img_size = sum(gray_counts)
        gray_counts[gray_counts < img_size * NOISE_PROPORTION] = img_size       # mark noise color (takse too little of the image)
        path_color = gray_colors[np.argsort(gray_counts)[0]]                                    # find the least common color
        path_size = gray_counts[np.argsort(gray_counts)[0]]
        background_color = gray_colors[np.argsort(gray_counts)[1]]

        # simple threshold, use the least frequent color (which should not be the background)
        thres = np.uint8(np.where(gray == path_color, 255, 0)) # produce binary image for the path color found
        input_shape = thres.shape   # threshold image shape (for creating mask)

        # compute Principle Components
        # to find line between two paths
        center1, pca_vector_1, pca_val = Path_PCA(thres)

        slice_dir = np.argmin(pca_val)  # slice by the minimum variance
        # Create the masks to separate two paths
        mask_one = np.ones(input_shape, dtype="uint8")                                   # generate mask
        mask_one = set_mask(mask_one, center1[:,0], pca_vector_1[:,slice_dir], 0)       # set above 0
        mask_two = np.zeros(input_shape, dtype="uint8")                                  # generate mask
        mask_two = set_mask(mask_two, center1[:,0], pca_vector_1[:,slice_dir], 1)       # set above 1
        # generate the two path segments
        bottom_path = cv2.bitwise_and(thres,thres,mask=mask_one)
        top_path = cv2.bitwise_and(thres,thres,mask=mask_two)

        # Compute Principle Components for both path segments (center point(mean), direction vector(eigvec), variance vector(eigval))
        path_center1, _, _ = Path_PCA(bottom_path)
        path_center2, _, _ = Path_PCA(top_path)

        # center of two segments (start location)
        bot_hori_cent, bot_vert_cent = int(path_center1[:,0][0]), int(path_center1[:,0][1])
        top_hori_cent, top_vert_cent = int(path_center2[:,0][0]), int(path_center2[:,0][1])
        path_direction = compute_slope((bot_hori_cent, bot_vert_cent),(top_hori_cent, top_vert_cent))

        # angle between the path direction and a set vector
        path_angle = compute_angle(FORWARD_DEFAULT,path_direction)

        #    [path_color, background_color, theta, size, location[x,y]]
        current_path_properties = [path_color, background_color, path_angle, path_size/img_size, bot_hori_cent, bot_vert_cent]
        print(current_path_properties)
        path_object.compareAndUpdate(current_path_properties)
        print(list(zip(path_object.PROPERTY_TAGS, path_object.properties)))

        # check the values with more examples
        if (path_object.properties[path_object.PROPERTY_TAGS.index("x")] < width/2):
            print("move left")
        else:
            print("move right")

        if (path_object.properties[path_object.PROPERTY_TAGS.index("bot2top_theta")] < np.pi/2):
            print("rotate right")
        else:
            print("rotate left")
