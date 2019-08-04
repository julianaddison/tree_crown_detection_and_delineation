# Import required packages
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import time
from scipy.spatial import distance
from sympy.geometry import Circle, Point
from scipy.stats import itemfreq
import os

# Check if cnt lies in boundary through OpenCV's pointpolygon test
def check_point(cnt, bound):
    # checks if the point exists within the contours including the boundary or outside
    check_sum = 0
    for i in range(0, len(bound)):
        # if window boundary points are within contour boundary or contour add one
        if cv2.pointPolygonTest(cnt, (bound[i][0], bound[i][1]), False) >= 0:
            check_sum = check_sum + 1
    return check_sum

# Find sliding window boundary e.g. using 3 pixel window returns 8 points
def find_boundary(x, y, windowSize):
    a = np.arange(x, x+windowSize+1) #10
    b = np.arange(y, y+windowSize+1) #200

    # find all pixel indices of square grid and convert to array
    cell = []
    for i in a:
        for j in b:
            cell.append([i, j])

    cell = np.array(cell)

    # find outer most pixel indices
    left = np.where(cell == x)[0] #10
    right = np.where(cell == y)[0] #25
    top = np.where(cell == (x+windowSize))[0] #200
    bottom = np.where(cell == (x+windowSize))[0] #255
    ind = np.concatenate((left, right, top, bottom), axis=0)

    cell = np.unique(cell[ind], axis=0)
    return cell

# Obtain similarity map
def sliding_window(img0, img1, stepSize, windowSize, box, cnt):
    start = np.min(box, axis=0)
    end = np.max(box, axis=0)

    # create rotated rect image mask
    image = np.zeros((img0, img1), np.uint8)
    # cv2.fillPoly(image, [box], 255)

    # slide a window across the image mask
    for y in range(start[0], end[0]-windowSize+1, stepSize):
        for x in range(start[1], end[1]-windowSize+1, stepSize):
            # check if window boundary is within contour
            # StartboundTime = time.time()
            ker = find_boundary(x=y, y=x, windowSize=windowSize)
            # EndboundTime = time.time()
            # print("Bound --- %s seconds ---" % (StartboundTime - EndboundTime))
            # ker = [[y, x], [y, x+windowSize], [y+windowSize, x], [y+windowSize, x+windowSize]]
            if check_point(cnt=cnt, bound=ker)/len(ker) >= 0.95: # set threshold at 80% OR == len(ker):
                # update window in image mask
                image[x:(x + windowSize), y:(y + windowSize)] = cv2.add(image[x:(x + windowSize), y:(y + windowSize)],
                                                                        1)
            # EndcheckTime = time.time()
            # print("Check --- %s seconds ---" % (EndboundTime - EndcheckTime))
    return image

# Find local maximas with iri_bri using peak_local_max
def find_peaks(mask, iri_bri):
    mask = img_as_float(mask)

    # image_max is the dilation of im with a 3 structuring element
    # it is used within peak_local_max function
    image_max = ndi.maximum_filter(mask, size=3, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    # peaks are separated by at least min_distance i.e. iri or bri
    coordinates = peak_local_max(mask, min_distance=iri_bri)

    return coordinates

# Find point centers for each contour from similarity map
def find_peak_centers(img0, img1, peak):
    # Create image mask
    image = np.zeros((img0, img1), np.uint8)

    # Set peak points with 255
    for i in peak:
        image[i[0], i[1]] = 255

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # Find contours
    #ret_peak, thresh_peak = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts_peak, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    cnts_centers = []
    #print(cnts_peak)
    for c in cnts_peak:

        #compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # update contour centers
        cnts_centers.append([cX, cY])

    #plt.imshow(image)
    return cnts_centers

# Find distance between two points
def find_pt_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Reorder peak centers based on a reference point
def order_pts(peak_centers, box):
    peak_centers_ordered = []
    tmp = peak_centers.copy()

    box = box.tolist()
    min_box = min(box)

    while len(tmp) > 0:
        if len(tmp) == len(peak_centers):
            nxt_pt = sorted(tmp, key=lambda e: find_pt_distance(e, min_box))[0]
            peak_centers_ordered.append(nxt_pt)
            tmp.remove(nxt_pt)

        nxt_pt = min(tmp, key=lambda e: find_pt_distance(e, nxt_pt))
        peak_centers_ordered.append(nxt_pt)
        tmp.remove(nxt_pt)

    return peak_centers_ordered

# Find point of intersection of circle to get perpendicular bisector
def circle_intersect(pt1, pt2, SE_radius):
    # find point of intersection of circle to get perpendicular bisector
    c1 = Circle(Point(pt1[0], pt1[1]), SE_radius)
    c2 = Circle(Point(pt2[0], pt2[1]), SE_radius)
    intersection = c1.intersection(c2)
    if len(intersection) == 1:
        intersection.append(intersection[0])
    p1 = intersection[0]
    p2 = intersection[1]
    xs1, ys1 = int(p1.x), int(p1.y)
    xs2, ys2 = int(p2.x), int(p2.y)

    return xs1, ys1, xs2, ys2

def find_coord(xs1, ys1, xs2, ys2, mid_x, mid_y, SE_radius, overlap=True):
    if overlap == True:
        if (xs1 - xs2) == 0:
            x1 = int(mid_x)
            x2 = int(mid_x)

            y1 = int(mid_y - SE_radius * 2)
            y2 = int(mid_y + SE_radius * 2)
        else:
            # find line through perpendicular bisector points
            m = (ys1 - ys2) / (xs1 - xs2)

            x1 = int(mid_x + np.sqrt(np.square(SE_radius * 2) / (1 + np.square(m))))
            x2 = int(mid_x - np.sqrt(np.square(SE_radius * 2) / (1 + np.square(m))))

            y1 = int(m * (x1 - mid_x) + mid_y)
            y2 = int(m * (x2 - mid_x) + mid_y)
    else:
        if (ys1 - ys2) == 0:
            x1 = int(mid_x)
            x2 = int(mid_x)

            y1 = int(mid_y - SE_radius * 2)
            y2 = int(mid_y + SE_radius * 2)
        else:
            # find line through perpendicular bisector points
            m = -(xs1 - xs2) / (ys1 - ys2)

            x1 = int(mid_x + np.sqrt(np.square(SE_radius * 2) / (1 + np.square(m))))
            x2 = int(mid_x - np.sqrt(np.square(SE_radius * 2) / (1 + np.square(m))))

            y1 = int(m * (x1 - mid_x) + mid_y)
            y2 = int(m * (x2 - mid_x) + mid_y)

    return x1, y1, x2, y2

# Delineate Large tree contours after finding tree centers
def delineate_cnt(img0, img1, peak_centers, SE_radius, box, cnt, area=800): # d=18
    # main mask where delineation lines will be drawn on
    mask = np.zeros((img0, img1), np.uint8)
    cv2.fillPoly(mask, [cnt], 255)

    # reorder points
    peak_centers = order_pts(peak_centers=peak_centers, box=box)

    # find distance to nearest neighbour point. Compute euclidean dist of peak centers and find lower diag values
    dst = distance.cdist(peak_centers, peak_centers, 'euclidean')
    dst = np.tril(dst)
    dst = np.array(([[dst[i+1][i], int(i), int(i+1)] for i in range(len(dst)-1)]))

    # OVERLAP POINTS - find all overlapping points where diameter > euclidean dist
    overlap_ind = np.where((dst[:, 0] <= (SE_radius*2)))
    for i in range(0, len(overlap_ind[0])):

        # select overlapping points
        pt1, pt2 = peak_centers[int(dst[overlap_ind][i][1])], peak_centers[int(dst[overlap_ind][i][2])]

        # find mid points
        mid_x = (pt2[0] + pt1[0])/2
        mid_y = (pt2[1] + pt1[1])/2

        # find point of intersection of circle to get perpendicular bisector
        xs1, ys1, xs2, ys2 = circle_intersect(pt1=pt1, pt2=pt2, SE_radius=SE_radius)
        x1, y1, x2, y2 = find_coord(xs1=xs1, ys1=ys1, xs2=xs2, ys2=ys2, mid_x=mid_x, mid_y=mid_y, SE_radius=SE_radius)

        # Step 1 - Create empty mask and draw polylines for contour and perpendicular bisector line
        # line mask thickness > 1 i.e. set at 2 here to prevent X type intersection
        maskLin = np.zeros((img0, img1), np.uint8)
        maskCnt = maskLin.copy()
        cv2.line(maskLin, (x1, y1), (x2, y2), 255, 2)
        cv2.polylines(maskCnt, [cnt], True, (255, 255, 255))
        cv2.line(maskCnt, (x1, y1), (x2, y2), 255, 1)

        # Step 2 - Perform bitwise_and operation to retain similar points
        diff = cv2.bitwise_and(maskLin, maskCnt)
        cnts_cut = np.where(diff > 0)
        cnts_cut = np.column_stack(cnts_cut[::-1])  # "-1" reverses order of columns

        # Step 3 - Split contour by using perpendicular line as mask on filled contour polygon and find contours
        cv2.line(mask, (cnts_cut[0][0], cnts_cut[0][1]), (cnts_cut[len(cnts_cut)-1][0], cnts_cut[len(cnts_cut)-1][1]),
                 0, 2)

    # NEAR OVERLAP POINTS - find near overlap points where diameter < euclidean dist < diameter*2
    near_overlap_ind = np.where((dst[:, 0] > (SE_radius * 2)) & (dst[:, 0] < (SE_radius * 2)*1.75))
    for i in range(0, len(near_overlap_ind[0])):
        pt1, pt2 = peak_centers[int(dst[near_overlap_ind][i][1])], peak_centers[int(dst[near_overlap_ind][i][2])]

        mid_x = (pt2[0] + pt1[0])/2
        mid_y = (pt2[1] + pt1[1])/2

        x1, y1, x2, y2 = find_coord(xs1=pt1[0], ys1=pt1[1], xs2=pt2[0], ys2=pt2[1], mid_x=mid_x, mid_y=mid_y,
                                    SE_radius=SE_radius, overlap=False)

        cv2.line(mask, (x1, y1), (x2, y2), 0, 2)

    # TODO: DISTANT CIRCLES IF IT OCCURS ALOT
    '''
    non_overlap_ind = np.where((dst > (SE_radius * 2)*1.5) & (dst != 0))
    non_overlap_ind = np.column_stack(non_overlap_ind)

    # Combine near_overlap_ind and overlap_ind -> make freq table -> find where freq is greater than 2
    # i.e. Circles have been addressed at both ends

    # generate freq table
    test = itemfreq(np.concatenate(np.row_stack([overlap_ind, near_overlap_ind])))
    test_ind = np.where(test[:, 1] >= 2)

    end_ind = np.where(dst == np.max(dst))

    # exclude anything with freq >= 2 AND exclude if any end point is present
    non_overlap_ind = np.delete(non_overlap_ind, np.where(non_overlap_ind == test[:, 0][test_ind]), axis=0)

    # select neighbour circle pairs with dist > diameter*1.5 = use min
    a = np.column_stack(np.where(dst != 0))
    b = np.ravel(dst[dst != 0])
    c = np.column_stack([a, b])

    d = []
    for i in range(0, len(peak_centers)):
        tmp = c[np.where(c[:, 0:2] == i)[0]]
        d.append(tmp[np.where(tmp == min(tmp[:, 2]))[0]])'''

    # Find new contours on mask
    cnts_break, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''for c in cnts_break:
        cv2.drawContours(img, c, -1, (255, 255, 255), 2)'''

    # Remove small contours i.e. those less than 40% of normal tree area
    cnts_area =[]
    for c in cnts_break:
        cnts_area.append(cv2.contourArea(c))

    cnts_area = np.array(cnts_area)
    cnts_area_ind = np.where(cnts_area > 0.4 * area)
    cnts_break = list(cnts_break[index] for index in cnts_area_ind[0])

    return cnts_break
