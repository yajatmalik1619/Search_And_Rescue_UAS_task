#importing the necessary libraries cv2 and numpy 
import cv2
import numpy as np

#setting the color bounds for ocean and land detecting in HSV, tuned initally through an approximate range
#and then refined through trial and error on multiple images 
#(as green on casualties and light blue on pad were being misclassified in a few cases)
ocean_color_lower_bound = (95, 80, 30) 
ocean_color_upper_bound = (130, 255, 160)
land_color_lower_bound = (35, 50, 80)
land_color_upper_bound = (85, 255, 150)

#function to the detect the shapes from the image using color segmentation and separating them as objects
#of interests later on 
def detect_shapes_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #creating masks for ocean and land using the predefined color bounds
    ocean_mask = cv2.inRange(hsv_image, ocean_color_lower_bound, ocean_color_upper_bound)
    land_mask = cv2.inRange(hsv_image, land_color_lower_bound, land_color_upper_bound)
    
    #collecting the ocean and land masks as one whole background to sepearate them from objects of interest
    background_mask = cv2.bitwise_or(ocean_mask, land_mask)
    #inverting the background mask to get the shapes mask
    shapes_mask = cv2.bitwise_not(background_mask)
    
    #using kernel(3,3) as (2,2) gave rise to some noise points in a few images
    kernel = np.ones((3,3), np.uint8)
    #to remove noise and smoothening shapes, using MORPH_OPEN 
    shapes_mask = cv2.morphologyEx(shapes_mask, cv2.MORPH_OPEN, kernel)
    #to fill small holes in shapes, using MORPH_CLOSE
    shapes_mask = cv2.morphologyEx(shapes_mask, cv2.MORPH_CLOSE, kernel)
    
    return shapes_mask, ocean_mask, land_mask

#function to segregate valid shapes and protect them by creating a protection mask
#protection is needed as some shapes faced coloring during the color grading step
def create_protection_mask(image, shapes_mask):
    #finding contours from the shapes mask
    contours, _ = cv2.findContours(shapes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #initalizing protection mask of the same dimensions as the image
    protection_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    #iterating through each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        #ignoring small contours (probably noise)
        if area < 50:
            continue
        
        #drawing filled contours on the protection mask
        cv2.drawContours(protection_mask, [contour], -1, 255, thickness=-1)
    
    #dilating the protection mask to ensure complete coverage
    kernel = np.ones((5,5), np.uint8)
    protection_mask = cv2.dilate(protection_mask, kernel, iterations=1)
    
    return protection_mask

#function to color grade the image (enhancement of color for visibility in ocean and land areas)
#also helps to remove excess noise by protecting the shapes 
def color_grade(image_path):
    image = cv2.imread(image_path)
    #preventive measure
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    #using detect_shapes_mask function to get the masks for shapes, ocean and land
    shapes_mask, ocean_mask, land_mask = detect_shapes_mask(image)

    #using create_protection_mask function to create a protection mask for the shapes
    protection_mask = create_protection_mask(image, shapes_mask)
    
    #to fill small holes in ocean and land masks after protecting the shapes (so no changes occur in shapes)
    kernel = np.ones((5,5), np.uint8)
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)
    
    #creating protected ocean and land masks and removing shapes' areas from them
    ocean_mask_protected = cv2.bitwise_and(ocean_mask, cv2.bitwise_not(protection_mask))
    land_mask_protected = cv2.bitwise_and(land_mask, cv2.bitwise_not(protection_mask))
    
    #making image copy to apply color grading to it
    new_image = image.copy()
    #applying color grading by changing color values in pixels of ocean and land areas
    new_image[ocean_mask_protected > 0] = [200, 100, 0]
    new_image[land_mask_protected > 0] = [60, 200, 255]
    
    return new_image, ocean_mask, land_mask, protection_mask

#function to get the colors of the shapes by using the median of the HSV values in the contours 
def get_shape_color(hsv_image, contour, cx, cy):
    #getting bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    #initalizing full mask for the contours as same dimensions of the image
    mask_full = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    #drawing filled contours on the mask
    cv2.drawContours(mask_full, [contour], -1, 255, -1)
    
    #eroding the mask to avoid boundary pixels (possible noise)
    kernel = np.ones((3,3), np.uint8)
    mask_eroded = cv2.erode(mask_full, kernel, iterations=2)
    
    #creating a circular mask at the center of the contour
    radius = int(min(w, h) * 0.3)
    if radius < 3:
        radius = 3
    mask_center = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.circle(mask_center, (cx, cy), radius, 255, -1)
    #taking the intersect of eroded mask and center mask to get final mask for shapes
    final_mask = cv2.bitwise_and(mask_eroded, mask_center)
    
    #getting pixels in the final mask
    pixels = hsv_image[final_mask > 0]
    
    #failsafe
    if len(pixels) == 0:
        return (0, 0, 0)
    #calculating median HSV values
    median_hsv = np.median(pixels, axis=0)
    
    return tuple(median_hsv)

#function to find contours and classify them based on shape and color
def find_contours(original_image, shapes_mask):
    #same procedure as before
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    contours, _ = cv2.findContours(shapes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #initalizing list to hold detected objects
    detected_objects = []
    
    #iterating through each contour to classify them
    for idx, contour in enumerate(contours):
        #taking area of the contour
        area = cv2.contourArea(contour)
        
        #discarding small area countours (possibly noise)
        if area < 50:
            continue
        
        #taking perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        #calculating the type of polygon
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)
        
        #calculating centroid of the contour
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) #x coordinate 
        cy = int(M["m01"] / M["m00"]) #y coordinate

        #getting the mean HSV color of the shape using the get_shape_color function
        mean_hue = get_shape_color(hsv_image, contour, cx, cy)
        #getting individual H, S, V values
        h, s, v = mean_hue[:3]
        
        #classifying shape and color using respective functions
        shape_type = classify_shape(vertices, area, contour, perimeter)
        color_class = classify_color(mean_hue)
        
        # print(f"\nContour {idx}:")
        # print(f"  Area: {area:.2f}")
        # print(f"  Vertices: {vertices}")
        # print(f"  Shape: {shape_type}")
        # print(f"  HSV: H={h:.1f}, S={s:.1f}, V={v:.1f}")
        # print(f"  Color: {color_class}")
        # print(f"  Center: {(cx, cy)}")
        
        #creating object dictionary to hold all relevant information
        obj = {'contour': contour,'centroid': (cx, cy),'shape': shape_type,'color': color_class,'vertices': vertices}
        #appending the object to detected objects list
        detected_objects.append(obj)
    
    return detected_objects

#classification of shapes 
def classify_shape(vertices, area, contour, perimeter):
    #calculating circularity (will be used to differentiate between circle and star)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    #getting bounding rectangle to calculate extent
    x, y, w, h = cv2.boundingRect(contour)
    #will be used to detect triangles misclassified as squares
    extent = area / (w * h) if (w * h) > 0 else 0
    
    #classifying based on number of vertices and other parameters
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        if extent < 0.75: #not enough area covered, so classified as triangle
            return 'triangle'
        return 'square'
    
    elif 5 <= vertices <= 10:
        if circularity < 0.65: #not circular enough, so classified as star
            return "star"
        else:
            return "circle"
    
    elif vertices > 10:
        if circularity > 0.75: #circular enough, so classified as circle
            return 'circle'
        else:
            return 'star'
    #failsafe
    return 'unknown'

#classification of colors of shapes
def classify_color(mean_hue):
    #getting individual H, S, V values
    h, s, v = mean_hue[:3]
    
    #classifying based on HSV values
    if s < 25:
        return 'grey'
    if (h < 12 or h > 168) and s > 40:
        return 'red'
    elif 20 <= h < 40 and s > 40:
        return 'yellow'
    elif 40 <= h < 85 and s > 40:
        return 'green'
    elif 100 <= h < 135 and s > 40:
        return 'blue'
    elif 135 <= h <= 160 and s > 15:
        return 'pink'
    #failsafe
    return 'unknown'

#to classify shapes as pads and casualties as well as to add showcase details along the shape
def draw_shapes_on_image(image, detected_objects):
    #generating a copy 
    output_image = image.copy()
    
    #iterating through each detected object
    for obj in detected_objects:
        center = obj['centroid']
        shape_type = obj['shape']
        color = obj['color']
        contour = obj['contour']
        #drawing different markers based on shape type
        if shape_type == 'circle':
            boundary_color = (0, 255, 0)
            cv2.circle(output_image, center, 7, (0, 0, 255), -1)
        else:
            boundary_color = (0, 0, 255)
            cv2.circle(output_image, center, 5, (0, 0, 255), -1)

        cv2.drawContours(output_image, [contour], -1, boundary_color, 2)
        #adding text labels for shape and color
        label = f"{shape_type}-{color}"
        cv2.putText(output_image,label,(center[0] - 40, center[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),1)
        cv2.putText(output_image,f"V:{obj['vertices']}",(center[0] - 20, center[1] + 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255,0),1)

    return output_image

#function to separate casualties and rescue pads from detected objects
def separate_casualties_and_pads(detected_objects):
    #initalizing dictionaries to hold casualties and rescue pads
    casualties = {}
    rescue_pads = {}
    #iterating through each detected object
    for obj in detected_objects:
        #classifying based on shape type
        if obj['shape'] == 'circle':
            rescue_pads[obj['centroid']] = [obj['shape'], obj['color']]
        elif obj['shape'] in ['star', 'triangle', 'square']:
            casualties[obj['centroid']] = [obj['shape'], obj['color']]
    return casualties, rescue_pads
