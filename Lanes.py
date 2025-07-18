
# %%
import cv2
import numpy as np # linear algebra
# Import the following to run a live detection from any opened window.
from PIL import ImageGrab
# These to edit the threshold while detecting. All for debugging.
import threading
import tkinter as tk
from functools import partial
import os
# Just for GTA
import time


# %%
import math

def grayscale(img):
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   
def canny(img, low_threshold, high_threshold):
    
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
   
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def draw_curves(image, left_coeffs):
    height, width = image.shape[:2]
    plot_y = np.linspace( int((height - 1)*0.45),height*0.79, int(height*0.34))

    # Calculate x values for left lane
    left_plot_x = left_coeffs[0] * plot_y ** 2 + left_coeffs[1] * plot_y + left_coeffs[2]

    # # Calculate x values for right lane
    # right_plot_x = right_coeffs[0] * plot_y ** 2 + right_coeffs[1] * plot_y + right_coeffs[2]

    # Stack the points for drawing
    left_points = np.array([np.transpose(np.vstack([left_plot_x, plot_y]))])
    # right_points = np.array([np.flipud(np.transpose(np.vstack([right_plot_x, plot_y])))])
    # points = np.hstack((left_points, right_points))
    points = np.hstack((left_points))
    # Draw the lanes
    cv2.polylines(image, np.int32([points]), isClosed=False, color=(0, 255, 255), thickness=10)

def merge_hough_lines(lines, threshold_angle=8):
    """
    Merge similar Hough lines by averaging their parameters.
    
    Args:
    - lines: List of Hough lines in the format [[x1, y1, x2, y2]]
    - threshold_angle: Maximum angle (in degrees) between lines to be considered similar
    
    Returns:
    - merged_lines: List of merged Hough lines in the same format as the input
    """
    merged_lines = []
    lines = np.array(lines)
    
    while len(lines) > 0:
        line = lines[0][0]  # Extracting the line from the extra layer of lists
        similar_lines_idx = []
        
        for i, other_line in enumerate(lines[1:], start=1):
            other_line = other_line[0]  # Extracting the line from the extra layer of lists
            angle_diff = np.abs(np.arctan2(line[3]-line[1], line[2]-line[0]) - np.arctan2(other_line[3]-other_line[1], other_line[2]-other_line[0])) * 180 / np.pi
            if angle_diff < threshold_angle:
                similar_lines_idx.append(i)
                
        if similar_lines_idx:
            similar_lines_idx.append(0)
            similar_lines = lines[similar_lines_idx][:,0]  # Extracting the lines from the similar indices
            merged_line = np.mean(similar_lines, axis=0, dtype=np.int32)
            merged_lines.append([merged_line.tolist()])
            lines = np.delete(lines, similar_lines_idx, axis=0)
        else:
            merged_lines.append([line.tolist()])
            lines = np.delete(lines, 0, axis=0)
    
    return merged_lines

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    f_lines = np.array([[0,0],[0,0]])
    poly_vertices = []
    try:
        merged_lines = merge_hough_lines(lines)

        draw_lines(line_img, lines)
#         print(merged_lines)
        line_img, poly_vertices = new_slope_lines(line_img,merged_lines)
    except:
        # In GTA unComment the following
#         rAll()
        cv2.putText(line_img, "Can't Detect lanes", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return line_img, poly_vertices

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    if img.shape[:2] != initial_img.shape[:2]:
        img = cv2.resize(img, (initial_img.shape[1], initial_img.shape[0]))
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    #lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges
## This for the RoI (Region of interest)
def get_vertices(image):
    """
    This function takes an image, crops the region and return the wanted part.
    How it works?
    We do have 6 points can be less can be more, but these points were kinda accurate in GTA window.
    From its name it points to correct posistion.
    How it looks? :                      mid left         w mid right
                                         /*----------------*\
                                        /                    \
                                       /                      \
                                      /                        \
                                     /                          \
                                    *                            *
                                    |                            |
                                    |                            |
                                    |                            |
                                    *----------------------------*
    These factors can't be more than one, 0.5 means the half of the image, whatever it was the rows or cols.
    ** Reminder: rows are Increasing downside (wnta nazl) w cols 3ady increasing -->
    For example 3ayz the left down corner [cols* 0, rows*1.0]
    atmna ttfhm
                                    
    """
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.00, rows*0.79]
    top_left     = [cols*0.00, rows*0.58]
    mid_left     = [cols*0.4375, rows*0.42]
    mid_right     = [cols*0.75, rows*0.42]
    bottom_right = [cols*1.0, rows*0.79]
    top_right    = [cols*1.0, rows*0.58] 
    ver = np.array([[bottom_left, top_left,mid_left,mid_right, top_right, bottom_right]], dtype=np.int32)
    # cv2.fillPoly(image, ver, color = (0,100,100))

    return ver

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary
def compareGray(im1, im2, tx1, tx2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(im1,cmap='gray')
    ax1.set_title(tx1, fontsize=50)
    ax2.imshow(im2,cmap='gray')
    ax2.set_title(tx2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# %%
# Lane finding Pipeline
def lane_finding_pipeline(image, s_thresh =(170, 255),
                          sx_thresh=(20, 255), kernel_size = 3,
                          h_threshold = 40, h_theta = 1* np.pi/180,
                          min_line_len= 150,max_line_gap = 3,
                          poly_vertices = []):
    
#     #Grayscale
#     gray_img = grayscale(image)
#     #Gaussian Smoothing
#     smoothed_img = gaussian_blur(img = gray_img, kernel_size = 3)
#     #Canny Edge Detection
#     compareGray(gray_img, smoothed_img,'Gray1',"Smoothed")
    canny_img = pipeline(image, s_thresh= s_thresh, sx_thresh=sx_thresh)
    #Masked Image Within a Polygon
    # compareGray(image,canny_img,'Orignal','S and SobleX')
    if poly_vertices:
        masked_img = select_regions(canny_img, poly_vertices)
    else :
        masked_img = region_of_interest(img = canny_img, vertices = get_vertices(image))
    smoothed_img = gaussian_blur(img = masked_img, kernel_size = kernel_size)
    masked_img_normalized = cv2.normalize(smoothed_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Display the normalized grayscale image
   # cv2.imshow('Masked_img', smoothed_img)    
    #Hough Transform Lines
    # compareGray(canny_img,masked_img,'S and SobleX', 'Masked')
    houghed_lines, vert = hough_lines(img = smoothed_img, rho = 1, theta =h_theta, threshold = h_threshold , min_line_len = min_line_len, max_line_gap = max_line_gap)
    # houghed_lines_normalized = cv2.normalize(houghed_lines, None, alpha = 0, beta= 255, norm_type= cv2.NORM_MINMAX, dtype= cv2.CV_8U)
    # cv2.imshow("Hough_lines", houghed_lines_normalized)
    #Draw lines on edges
    # compareGray(masked_img,houghed_lines,'Masked','Hough')
    houghed_with_main = draw_main_line(houghed_lines)
    output = weighted_img(img = houghed_with_main, initial_img = image, α=0.8, β=1., γ=0.)
    
    return output, vert

# %% [markdown]
# # Helping Functions 

# %%
def generate_fake_lane(expected_lane_width, img):
    # Assuming center of the image as the starting point
    _, image_width = img.shape[:2]
    center_x = image_width / 2
    
    # Assuming a fixed lane width and symmetrical lanes
    half_lane_width = expected_lane_width / 2
    
    # Generate fake left and right lane lines
    fake_left_slope = 0
    fake_left_intercept = center_x - half_lane_width
    fake_right_slope = 0
    fake_right_intercept = center_x + half_lane_width
    
    return (fake_left_slope, fake_left_intercept), (fake_right_slope, fake_right_intercept)


# %%
def draw_main_line(image):
    image_height, image_width = image.shape[:2]
    # Angle of the line in degrees
    angle_deg = 128
    
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Calculate the slope of the line
    slope = np.tan(angle_rad)
    
    # Intercept on the y-axis
    y_intercept = image_height
    
    # Calculate x-coordinate for the point at y = image_height
    x_at_y_max = 700
    
    # Calculate the corresponding y-coordinate
    y_at_x_max = (image_height / 2) - 50
    
    # Draw the line
    return cv2.line(image, (x_at_y_max, image_height), (int(image_width / 2), int(y_at_x_max)), (0, 0, 255), thickness=2)


# %%
def are_lines_touching(line1, line2, tolerance):
    """
    Checks if two lines are close enough to be considered touching.
    
    Args:
      line1: An array [x1, y1, x2, y2] representing the first line.
      line2: An array [x1, y1, x2, y2] representing the second line.
      tolerance: The maximum distance between the lines to be considered touching.
    
    Returns:
      True if the lines are considered touching, False otherwise.
    """
    
    # Calculate the minimum distance between the lines.
    # There are different approaches to achieve this, here we use the distance formula.
    [x1_1, y1_1, x2_1, y2_1] = line1[0]
    [x1_2, y1_2, x2_2, y2_2] = line2[0]
    # Find the closest point on line1 to line2
    # https://stackoverflow.com/a/5942802
    dx1, dy1 = x2_1 - x1_1, y2_1 - y1_1
    dx2, dy2 = x2_2 - x1_2, y2_2 - y1_2
    fact = 2
    if(abs(dx1) <  tolerance //fact or abs(dx2) <tolerance // fact or abs(x1_1 - x2_2) < tolerance // fact or abs(x1_2 - x2_1) < tolerance // fact):
        return True
    t = (dx1 * (x1_2 - x1_1) + dy1 * (y1_2 - y1_1)) / (dx1 * dx2 + dy1 * dy2)
    
    if t < 0:
        projection_x = x1_1
        projection_y = y1_1
    elif t > 1:
        projection_x = x2_1
        projection_y = y2_1
    else:
        projection_x = x1_1 + t * dx1
        projection_y = y1_1 + t * dy1

    # Check if the projection point is within the line segments of both lines
    min_x1, max_x1 = min(x1_1, x2_1), max(x1_1, x2_1)
    min_y1, max_y1 = min(y1_1, y2_1), max(y1_1, y2_1)
    min_x2, max_x2 = min(x1_2, x2_2), max(x1_2, x2_2)
    min_y2, max_y2 = min(y1_2, y2_2), max(y1_2, y2_2)
    
    if not (min_x1 <= projection_x <= max_x1 and min_y1 <= projection_y <= max_y1 and
          min_x2 <= projection_x <= max_x2 and min_y2 <= projection_y <= max_y2):
        return False
    
    # Calculate thetas for both lines using the projection point
    theta1 = ((projection_x - x1_1) * (x2_1 - x1_1) +
              (projection_y - y1_1) * (y2_1 - y1_1)) / ((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
    theta2 = ((projection_x - x1_2) * (x2_2 - x1_2) + 
              (projection_y - y1_2) * (y2_2 - y1_2)) / ((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)
    theta_lower_threshold = 0 - tolerance/200
    theta_higher_threshold = 1 + tolerance/200
  # Check if both thetas are within the allowed range (0 <= theta <= 1)
    return theta_lower_threshold <= theta1 <= theta_higher_threshold and theta_lower_threshold <= theta2 <= theta_higher_threshold

# %%
def are_lines_touching_in_array(lines, tolerance):
    """
    Checks if any pair of lines in the array are touching.
    
    Args:
      lines: A list of tuples (x1, y1, x2, y2) representing Hough lines.
      tolerance: The maximum distance between the lines to be considered touching.
    
    Returns:
      True if any pair of lines is touching, False otherwise.
    """
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if are_lines_touching(lines[i], lines[j], tolerance):
                return True
    return False

# %%
def find_touching_lines(lines, tolerance = 10):
    """
    Finds groups of touching lines in an array of Hough lines.
    
    Args:
      lines: A list of tuples (x1, y1, x2, y2) representing Hough lines.
      tolerance: The maximum distance between the lines to be considered touching.
    
    Returns:
      A list of lists, where each inner list represents a group of touching lines.
    """

    # Create a dictionary to store lines and their associated touching lines.
    touching_lines = {}
    for i in range(len(lines)):
        touching_lines[i] = []

    # Iterate through all pairs of lines and check if they are touching.
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if are_lines_touching(lines[i], lines[j], tolerance):
                touching_lines[i].append(j)
                touching_lines[j].append(i)

    # Find connected components (groups of touching lines) using DFS.
    def dfs(i, visited, group):
        visited.add(i)
        group.append(lines[i])
        for neighbor in touching_lines[i]:
            if neighbor not in visited:
                dfs(neighbor, visited, group)

    groups = []
    visited = set()
    for i in range(len(lines)):
        if i not in visited:
            group = []
            dfs(i, visited, group)
            groups.append(group)
    # print(f"input to touching lanes was {len(lines)}, output{len(groups)}")
    return groups

# %%
def draw_many_lines(lines, image, color):
    cv2.putText(image,"Drawing many lines",(250,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    # print(f"This is the color: {color}, and these are the lines: {lines}")
    for group_lines in lines:
        if isinstance(group_lines[0], list):
            for line in group_lines:
                for segment in line:
                    x1, y1, x2, y2 = segment
                    cv2.line(image, (x1, y1), (x2, y2), color, 10)
        else:
            try:
                [[x1, y1, x2, y2]] = group_lines
            except :
                [x1, y1, x2, y2] = group_lines
            cv2.line(image, (x1, y1), (x2, y2), (100,100,100), 10)


# %%
def calculate_lane_width(line_array, image):
    rows, cols = image.shape[:2]
    # Extract slope and intercept for each lane
    slope1, intercept1 = line_array[0]
    slope2, intercept2 = line_array[1]

    # Calculate x-coordinate where each lane starts (assuming it starts at y=0)
    lane1_beginning = int((rows*0.79 - intercept1) / slope1)
    lane2_beginning = int((rows*0.79 - intercept2) / slope2)
    # Calculate the width between the lanes
    lane_width = abs(lane2_beginning - lane1_beginning)
    # print(f"width = {lane1_beginning} - {lane2_beginning} = {lane_width}")

    return lane_width

# %%
def remove_large_values(arr, tolerance=0.2):
    removed_indices = set()

    for i in range(len(arr)):
        if i in removed_indices:
            continue

        for j in range(i+1, len(arr)):
            if j in removed_indices:
                continue

            sum_val = arr[i] + arr[j]
            for k in range(len(arr)):
                if k == i or k == j or k in removed_indices:
                    continue

                if arr[k] > sum_val * (1 - tolerance):
                    removed_indices.add(k)
    if removed_indices:
        return removed_indices

# %%
def xy_from_slope(line, rows):
    if(len(line) > 1):
        slope, intercept = line
    else:
        slope, intercept = line[0]
    y1 = int(rows * 0.42)
    y2 = int(rows * 0.79)
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return [[x1,y1,x2,y2]]

# %%
def slope_from_xy(line):
    if(len(line) > 1):
        x1,y1,x2,y2 = line
    else :
        x1,y1,x2,y2 = line[0]
    # No vertical or almost vertical lanes
    if abs(x1-x2) > 10:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        return [m,c]

# %%
def mean_slope(lines):
    ar = []
    for line in lines:
        ar.append(slope_from_xy(line))
    return np.mean(ar, axis = 0)

# %%
def generate_color(index):
    binary = bin(index)[2:].zfill(3)  # Convert index to binary with 3 digits
    r = int(binary[0]) * 255
    g = int(binary[1]) * 255
    b = int(binary[2]) * 255
    return (r, g, b)

# %%
def detect_lanes(touching_lines, img, old = False):
#     print(f"length :{len(touching_lines)}")
    order = [0,1,3,2]
    right_found = False
    poly_vertices = []
    rows, cols = img.shape[:2]
    center_lane_width = 1000
    center_polys = []
    center_found = False
    if(len(touching_lines) >= 2):
        cv2.putText(img, f"{len(touching_lines)} different touching lanes", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        slopes = []
        if old :
            slopes = touching_lines
        else :
            for i,lines in enumerate(touching_lines) :
                mean = mean_slope(lines)
                slopes.append(mean)
                # draw_many_lines(np.array([xy_from_slope(mean,rows)]), img)
                color = generate_color(i+1)
                draw_many_lines(np.array(lines) ,img, color)
            
        # Calculate lane width for each pair of lines and check if it falls within the range
        matching_pairs_init = []
        width_diff = [] 
        for i in range(len(slopes)):
            for j in range(i + 1, len(slopes)):
                # To prevent al8baaaaaaaa 
                # slope[i] = line 1,, slope[j] for line 2,,, Each conatins slope, interception
                line_array = np.array([slopes[i], slopes[j]])  
                lane_width = calculate_lane_width(line_array, img)
                # print(f" width {lane_width}")
                
                if 500  <= lane_width <= 1000:
                    cv2.putText(img, "good width ", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 2)
                    matching_pairs_init.append(line_array)
                    width_diff.append(lane_width)
                else :
                    cv2.putText(img,f"small or big lane width{lane_width}" ,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 2)
        if(width_diff):
            removed_indices = remove_large_values(width_diff)
            matching_pairs = [val for i, val in enumerate(matching_pairs_init) if i not in removed_indices]
            # Check if there are any matching pairs
            if len(matching_pairs) != 0:
                # print("No pairs of lines found with lane width in the desired range.")
                # Find the best match in the desired order [left_line, right_line]
                cv2.putText(img, "good lanes", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 100), 2)
                # print("Start with an empty poly_vertices")
                poly_vertices = []
                zz= 0
                tol = 0
                for l1, l2 in matching_pairs:
                    for slope, intercept in [l1, l2]:
                        zz += 1
                        y1 = int(rows * 0.5)
                        y2 = int(rows * 0.79)
                        x1=int((y1-intercept)/slope)
                        x2=int((y2-intercept)/slope)
                        poly_vertices.append((x1, y1))
                        poly_vertices.append((x2, y2))
                        # draw_lines(img, np.array([[[x1,y1,x2,y2]]]))
                        if zz % 2 ==0:
                            poly = [poly_vertices[i+(zz-2)*2] for i in order]
                            left_midpoint = ((poly[0][0] + poly[3][0]) // 2, (poly[0][1] + poly[3][1]) // 2)
                            right_midpoint = ((poly[2][0] + poly[1][0]) // 2, (poly[2][1] + poly[1][1]) // 2)
                            dotted_line_pattern = 20  # Adjust the pattern as needed (length of dash + length of gap)
                            line_color = (255//zz, 255//zz, 100)  # Adjust the color as needed
                            
                            num_dots = int(np.linalg.norm(np.array(right_midpoint) - np.array(left_midpoint)) / dotted_line_pattern)
                            for kl in range(num_dots + 1):
                                if num_dots == 0:
                                    num_dots = 3
                                alpha = kl / num_dots
                                dot_position = (
                                    int((1 - alpha) * left_midpoint[0] + alpha * right_midpoint[0]),
                                    int((1 - alpha) * left_midpoint[1] + alpha * right_midpoint[1])
                                )
                                cv2.circle(img, dot_position, 5, line_color, -1)
                                if kl == 0 :
                                    tol = dot_position[0] - cols*0.5
                                    print(f"The center Deviation is {tol}")

                                    if l1[0] > 0 and l2[0] > 0:
                                        right_found = True
                                        cv2.fillPoly(img, pts = np.array([poly],'int32'), color = (0,0,200))
                                    elif (l1[0] * l2[0]) < 0 :
                                        lane_width = calculate_lane_width([l1, l2], img)
                                        if lane_width < center_lane_width :
                                            center_lane_width = lane_width
                                            center_polys = np.array([poly],'int32')
                                            center_found = True
                                    else :
                                        cv2.fillPoly(img, pts = np.array([poly],'int32'), color = (255,0,0))
                                        # front lane found
                if center_found:
                    cv2.fillPoly(img, pts = center_polys, color = (0,255,0))
                cv2.putText(img, f"Now poly_vertices has {len(poly_vertices)}", (50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 100), 2)
                # print(f"Now poly_vertices has {len(poly_vertices)}")
                # print(f"they are :{poly_vertices}")
                if(right_found):
                    cv2.putText(img, "Oh shit right found ", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    ## In GTA Un comment the following
                    # if tol > 0 : 
                    #     PressKey(W)
                    #     PressKey(D)
                    # elif tol < 0:
                    #     PressKey(W)
                    #     PressKey(A)
                    # elif tol != 0 :
                    #     PressKey(W)

    return poly_vertices

# %%
def select_regions(image, poly_vertices, line_thickness=30):
    # print(f"Selecting Region with {len(poly_vertices)} vertices. These vertices are: {poly_vertices}")
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    for i in range(0, len(poly_vertices), 2):
        line1_vertices = poly_vertices[i:i+2]

        if len(line1_vertices) != 2 :
            print("Invalid number of vertices for a line. Skipping...")
            continue

        # Draw the line on the image
        cv2.line(mask, tuple(line1_vertices[0]), tuple(line1_vertices[1]), ignore_mask_color, thickness=line_thickness)
        cv2.putText(mask, f"{tuple(line1_vertices[0])}",tuple(line1_vertices[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,100,100), 2)
        cv2.putText(mask, f"{tuple(line1_vertices[1])}",tuple(line1_vertices[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,100,100), 2)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, mask)
    # compareGray(image, mask, "Image", "Cropped")  # Assuming compareGray function exists
    # for i in range(0, len(poly_vertices), 2):
    #     line1_vertices = poly_vertices[i:i+2]
    #     if len(line1_vertices) != 2 :
    #         print("Invalid number of vertices for a line. Skipping...")
    #         continue
    #     # Draw the line on the image
    #     cv2.putText(masked_image, f"{tuple(line1_vertices[0])}",tuple(line1_vertices[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, ignore_mask_color//2, 2)
    #     cv2.putText(masked_image, f"{tuple(line1_vertices[1])}",tuple(line1_vertices[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, ignore_mask_color //2, 2)

    return masked_image

# %%
last_left_line = np.array([0,0])
last_good_left = last_good_right = last_right_line = np.array([0,0])
ltimes = 0
times = 0

def new_slope_lines (image, lines): 
    global last_left_line, last_right_line, times, ltimes, last_good_left, last_good_right
    # let's Prepare Our lines
    left_lines = []
    right_lines = []
    threshold_m = 0.03
    threshold_c = 20
    img = image.copy()
    rows, cols = image.shape[:2]
    poly_vertices = []
    left_points = []
    right_points = []
    included_lines = []
    fake = False
    all_one_side = False
    for line in lines :
        for x1,y1,x2,y2 in line :
            # No vertical or almost vertical lanes
            if abs(x1-x2) > 10:
                [m, c] = slope_from_xy([[x1,y1,x2,y2]])
                if m < -0.2 :
                    included_lines.append(line)
                    left_lines.append((m,c))
                   
                elif m > 0.2 :
                    included_lines.append(line)
                    right_lines.append((m,c))

    left_line = np.mean(left_lines, axis = 0)
    right_line = np.mean(right_lines, axis = 0)
    touching_lines = find_touching_lines(included_lines)
    if(left_lines):
#         touching_lines.extend([xy_from_slope(left_line,rows)])
        
#         print("###There are left and right")
        # if ((last_left_line[0] != left_line[0]) and (last_right_line[0] != right_line[0])):
        #     times = 0
        #     ltimes = 0
    # Escape small changes
        delta_ml = abs(last_left_line[0] - left_line[0])
        delta_cl = abs(last_left_line[1] - left_line[1])
        # if (delta_ml != 0 and delta_cl != 0):
        flagged = False
        if(delta_ml < threshold_m) :
            flagged = True
            cv2.putText(img, f"small change with left m :{round(delta_ml,3)} and C :{round(delta_cl,1)}", (400,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            left_line[0] = (last_left_line[0])
        if (delta_cl< threshold_c) :
            flagged = True

            left_line[1] = (last_left_line[1])
        elif not flagged :
            cv2.putText(img, "Left lane updated ", (200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 2)
            # print("Left lane updated")
            last_left_line = left_line

#     elif last_left_line[0] != 0:
#         touching_lines.extend([xy_from_slope(last_left_line,rows)])
    if(right_lines):
        delta_cr = abs(last_right_line[1] - right_line[1])
        delta_mr = abs(last_right_line[0] - right_line[0])
        # if (delta_mr != 0 and delta_cr != 0):
        flagged = False
        if(delta_mr < threshold_m ) :
            flagged = True
            right_line[0] = (last_right_line[0])
            cv2.putText(img, f"small change with m :{round(abs(last_right_line[0] - right_line[0]),3)} and C :{round(abs(last_right_line[1] - right_line[1]),1)}", (400,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if (delta_cr< threshold_c ) :
            flagged = True
            right_line[1] = (last_right_line[1])
        elif not flagged :
            cv2.putText(img, "Right lane updated ", (500,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 2)
            # print("Right lane updated")
            last_right_line = right_line
#         touching_lines.extend([xy_from_slope(right_line,rows)])

#     elif last_right_line[0] != 0:
#         touching_lines.extend([xy_from_slope(last_right_line,rows)])
    poly_vertices = detect_lanes(touching_lines, img)
    # if not fake :
    #     print(f"Not fake poly: {len(poly_vertices)}")
       ## IN GTA uncomment
        # if(abs(tol) >=50) : 
        #     if tol > 0 : 
        #         PressKey(W)
        #         PressKey(D)
        #     else :
        #         PressKey(W)
        #         PressKey(A)
        # elif tol != 0 :
        #     PressKey(W)
    return cv2.addWeighted(image,0.7,img,0.4,0.), poly_vertices
    # current_x, current_y = pyautogui.position()
    # # Move the cursor to the new position
    # pyautogui.moveTo(current_x, current_y + 1, duration=0)
    # print(f"fake: {len(poly_vertices)}")

    # rAll()
    # return cv2.addWeighted(image,0.7,img,0.4,0.), []
    

# %% [markdown]
# ## **Shortcomings in Live Video Stream**  
# Top left 800x600 window will be treated as the live stream u can adjust the window size from the line 25 in the second cell right after this one.
# w brdo u will find a comment just for indication.
# 
# Additionally, These parameters are being used with try and error, you will notice there are 2 buttons to use, one for the hard detection which works fine and almost accurate in the morning, can't be used in the night view; however, for the smooth one, ez parameters to detect any lanes. 
# 
# There is no Optimum value!
# 
# Change anything except the h_theta, I still can't understand how tf it works. 
# max_line_gap: Better to have it in the range between 0 : 5, more than 5 10 htb2a bt5re w tgm3 noise w t2ole edge.
# 

# %%
def get_hard():
    # Define parameters for daytime
    s_thresh_min = 150
    s_thresh_max = 255
    sx_thresh_min = 11
    sx_thresh_max = 150
    kernel_size = 15
    h_threshold = 40
    h_theta = 1 * np.pi / 180
    min_line_len = 70
    max_line_gap = 5
    return s_thresh_min, s_thresh_max, sx_thresh_min, sx_thresh_max, kernel_size, h_threshold, h_theta, min_line_len, max_line_gap

def get_smooth():
    # Define parameters for nighttime
    s_thresh_min = 140
    s_thresh_max = 180
    sx_thresh_min = 10
    sx_thresh_max = 150
    kernel_size = 5
    h_threshold = 40
    h_theta =  np.pi / 180
    min_line_len = 50
    max_line_gap = 10
    return s_thresh_min, s_thresh_max, sx_thresh_min, sx_thresh_max, kernel_size, h_threshold, h_theta, min_line_len, max_line_gap

# %%
def update_parameters(s_thresh_min_var, s_thresh_max_var, sx_thresh_min_var, sx_thresh_max_var, kernel_size_var, h_threshold_var, h_theta_var, min_line_len_var, max_line_gap_var):
    # Update parameters based on user input
    s_thresh = (s_thresh_min_var.get(), s_thresh_max_var.get())
    sx_thresh = (sx_thresh_min_var.get(), sx_thresh_max_var.get())
    kernel_size = kernel_size_var.get()
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    h_threshold = h_threshold_var.get()
    h_theta = h_theta_var.get() * np.pi / 180  # Convert to radians
    min_line_len = min_line_len_var.get()
    max_line_gap = max_line_gap_var.get()
    return s_thresh, sx_thresh, kernel_size, h_threshold, h_theta, min_line_len, max_line_gap     
def start_lane_detection(s_thresh_min_var, s_thresh_max_var, sx_thresh_min_var, sx_thresh_max_var, kernel_size_var, h_threshold_var, h_theta_var, min_line_len_var, max_line_gap_var):
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)
    last_vert = []
    counter = 0
    flagged = False
    last_time = time.time()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        #print(time.time() - last_time)
        #last_time = time.time() 
        _,frame = cap.read()
        screen = frame
        s_thresh, sx_thresh, kernel_size, h_threshold, h_theta,min_line_len, max_line_gap = update_parameters(
            s_thresh_min_var, s_thresh_max_var, sx_thresh_min_var, sx_thresh_max_var, kernel_size_var, h_threshold_var, h_theta_var, min_line_len_var, max_line_gap_var)
        
        op, vert = lane_finding_pipeline(screen, s_thresh, sx_thresh, kernel_size, h_threshold, h_theta, min_line_len, max_line_gap, [])
        if vert:
            flagged = True
            counter = 0
            last_vert = vert
            # update_parameters_for_smooth()
            cv2.putText(op, "Smoothed", (300,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 2)

        else :
            # print("A7a bgd")
            counter += 1
            if (counter >= 6):
                counter = 0
                if last_vert:
                    #print("Hard")
                    flagged = False
                    # update_parameters_for_hard()
                    last_vert = []
                elif flagged :
                    # update_parameters_for_smooth()
                    # print("Smoothed")
                    cv2.putText(op, "A7a Smoothed", (300,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 2)
        # cv2.imshow('window', cv2.cvtColor(draw_main_line(op), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
def update_parameters_for_hard():
    # Get hard condition parameters and update GUI sliders
    params = get_hard()
    s_thresh_min_var.set(params[0])
    s_thresh_max_var.set(params[1])
    sx_thresh_min_var.set(params[2])
    sx_thresh_max_var.set(params[3])
    kernel_size_var.set(params[4])
    h_threshold_var.set(params[5])
    h_theta_var.set(params[6])
    min_line_len_var.set(params[7])
    max_line_gap_var.set(params[8])

def update_parameters_for_smooth():
    # Get smooth condition parameters and update GUI sliders
    params = get_smooth()
    s_thresh_min_var.set(params[0])
    s_thresh_max_var.set(params[1])
    sx_thresh_min_var.set(params[2])
    sx_thresh_max_var.set(params[3])
    kernel_size_var.set(params[4])
    h_threshold_var.set(params[5])
    h_theta_var.set(params[6])
    min_line_len_var.set(params[7])
    max_line_gap_var.set(params[8])


root = tk.Tk()
root.title("Lane Detection Parameters")

window_width = 250
window_height = 700

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the x and y coordinates for the window to be centered
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

# Set the window's position
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

s_thresh_min_var = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="S Threshold Min")
s_thresh_min_var.set(150)
s_thresh_min_var.pack()

s_thresh_max_var = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="S Threshold Max")
s_thresh_max_var.set(255)
s_thresh_max_var.pack()

sx_thresh_min_var = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Sx Threshold Min")
sx_thresh_min_var.set(11)
sx_thresh_min_var.pack()

sx_thresh_max_var = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Sx Threshold Max")
sx_thresh_max_var.set(255)
sx_thresh_max_var.pack()

kernel_size_var = tk.Scale(root, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size (Odd)")
kernel_size_var.set(15)
kernel_size_var.pack()

h_threshold_var = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="H Threshold")
h_threshold_var.set(40)
h_threshold_var.pack()

h_theta_var = tk.Scale(root, from_=1, to= 180, orient=tk.HORIZONTAL, label="H Theta")
h_theta_var.set(1* np.pi/180)
h_theta_var.pack()

min_line_len_var = tk.Scale(root, from_=0, to=500, orient=tk.HORIZONTAL, label="Min Line Length")
min_line_len_var.set(75)
min_line_len_var.pack()

max_line_gap_var = tk.Scale(root, from_=0, to=50, orient=tk.HORIZONTAL, label="Max Line Gap")
max_line_gap_var.set(5)
max_line_gap_var.pack()


start_lane_detection_func = partial(start_lane_detection, s_thresh_min_var, s_thresh_max_var, sx_thresh_min_var, sx_thresh_max_var, kernel_size_var,
                                    h_threshold_var, h_theta_var, min_line_len_var, max_line_gap_var)

start_btn = tk.Button(root, text="Start Lane Detection", command=lambda: threading.Thread(target=start_lane_detection_func).start())
start_btn.pack()
hard_btn = tk.Button(root, text="Hard Conditions", command=update_parameters_for_hard)
hard_btn.pack()

smooth_btn = tk.Button(root, text="Smooth Conditions", command=update_parameters_for_smooth)
smooth_btn.pack()
root.mainloop()
