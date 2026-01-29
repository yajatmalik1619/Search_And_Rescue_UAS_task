#importing the necessary libraries along with pre_processing module 
import cv2
import numpy as np
import pre_processing as pp 

#defining priority mappings for age, emergency level, and camp capacity
age_priority = { "star": 3,"triangle": 2,"square": 1}
emergency_priority = {"red": 3,"yellow": 2,"green": 1}
camp_capacity = {"blue": 4,"pink": 3,"grey": 2}

#the main task, to calculate priority and assigning casualties to rescue pads accordingly
def calculate_priority(casualties, rescue_pads):
    #initializing a list to hold all priority scores
    all_priorities = []
    
    #iterating through each rescue pad 
    for pad_centroid, pad_info in rescue_pads.items():
        #extracting pad color
        pad_color = pad_info[1]
        #iterating through each casualty
        for casualty_centroid, casualty_info in casualties.items():
            #extracting the information of casualty
            casualty_shape, casualty_color = casualty_info
            age = age_priority[casualty_shape]
            emergency = emergency_priority[casualty_color]
            #calculating distance between pad and casualty
            distance = np.linalg.norm(np.array(pad_centroid) - np.array(casualty_centroid))
            #calculating priority score based on the weights calculated 
            priority_score = 7*emergency +3*age - 0.01*distance

            #adding all relevant information to the priority list
            all_priorities.append({
                'pad': pad_centroid,
                'casualty': casualty_centroid,
                'priority_scale': priority_score,
                'distance': distance,
                'emergency': emergency_priority[casualty_color],
                'age': age_priority[casualty_shape],
                'pad_color': pad_color,
            })
    #sorting the priorities in descending order based on priority score
    all_priorities.sort(key=lambda x: -x['priority_scale'])
    #initializing dictionaries to hold assignments and casualty colors per pad
    pad_assignments = {pad: [] for pad in rescue_pads.keys()}
    pad_casualty_colors = {info[1]: [] for info in rescue_pads.values()}
    #set to keep track of already assigned casualties so that no casualty is assigned more than once
    assigned_casualties = set()
    
    #iterating through the sorted priorities to assign casualties to pads
    for entry in all_priorities:
        #extracting relevant information from each entry
        pad = entry['pad']
        casualty = entry['casualty']
        pad_color = entry['pad_color']
        emergency_level = entry['emergency']
        age_level = entry['age']
        distance = entry['distance']
        
        #checking if casualty is already assigned
        if casualty in assigned_casualties:
            continue
        
        #checking if the pad has capacity to take more casualties
        if len(pad_assignments[pad]) < camp_capacity[pad_color]:
            #assigning relevant information to the dictionaries
            pad_assignments[pad].append(casualty)
            pad_casualty_colors[pad_color].append([age_level, emergency_level,distance])
            #keeping track of assigned casualties
            assigned_casualties.add(casualty)
    return pad_assignments, pad_casualty_colors

#drawing arrows for better visualization of assignments
def draw_assignment_arrows(image, pad_assignments):
    output_image = image.copy()

    #defining colors for arrows based on pad colors
    #arrow_colors = {'blue': (255, 0, 0), 'pink': (203, 192, 255), 'grey': (128, 128, 128)}
    
    #using black as it has better visibility on all pad colors
    arrow_color = (0,0,0)

    #iterating through each pad and its assigned casualties
    for pad_centroid, assigned_casualties in pad_assignments.items():

        #getting color of pad and arrow color for it respectively
        #pad_color = rescue_pads[pad_centroid][1]
        #arrow_color = arrow_colors.get(pad_color, (0, 0, 0))

        #drawing arrows from pad to each assigned casualty
        for casualty_centroid in assigned_casualties:
            cv2.arrowedLine(output_image, pad_centroid, casualty_centroid, arrow_color, 2, tipLength=0.03)
    
    return output_image

#function to print the assignments in a readable format
def print_assignments(pad_assignments, casualties, rescue_pads):
    #iterating through each pad and its assigned casualties
    for pad_centroid, assigned_casualties in pad_assignments.items():
        #getting pad color and capacity
        pad_color = rescue_pads[pad_centroid][1]
        capacity = camp_capacity[pad_color]
        
        # print(f"\n{pad_color.upper()} Rescue Pad at {pad_centroid}")
        # print(f"Capacity: {len(assigned_casualties)}/{capacity}")
        
        # for detailed information about each casualty
        if assigned_casualties:
            for casualty_centroid in assigned_casualties:
                shape, color = casualties[casualty_centroid]
                age_group = {'star': 'Child', 'triangle': 'Elderly', 'square': 'Adult'}[shape]
                emergency = {'red': 'Severe', 'yellow': 'Mild', 'green': 'Safe'}[color]
                priority = age_priority[shape] * emergency_priority[color]
                
                print(f"  - {age_group} ({shape}) - {emergency} ({color}) | Priority: {priority} | at {casualty_centroid}")
        else:
            print("  (No casualties assigned)")

#defining functions for score sum total and based on colors and their average based on base scoring system
def scores_sum_simple(pad_casualty_colors):
    #initializing score variables
    total_score = 0
    blue_score = 0
    pink_score = 0
    grey_score = 0
    #initializing count for average calculation
    count = 0

    #iterating through each pad color and its casualties
    for pad_color, casualties in pad_casualty_colors.items():
        #iterating through each casualty's details
        for age_level, emergency_level, distance in casualties:
            #calculating score based on age and emergency level
            score = age_level*emergency_level
            #increasing count
            count += 1
            #adding to total and respective color scores
            total_score += score
            if pad_color == 'blue':
                blue_score += score
            elif pad_color == 'pink':
                pink_score += score
            elif pad_color == 'grey':
                grey_score += score
    #calculating average score
    average_score = total_score / count if count > 0 else 0
    return total_score, average_score, blue_score, pink_score, grey_score

#function for the same based on the scoring system used
def scores_sum_next(pad_casualty_colors):
    #initializing score variables
    total_score = 0
    blue_score = 0
    pink_score = 0
    grey_score = 0
    count = 0
    #iterating through each pad color and its casualties
    for pad_color, casualties in pad_casualty_colors.items():
        for age_level, emergency_level, distance in casualties:
            #calculating score based on the defined scoring system
            score = 7*emergency_level + 3*age_level - 0.01*distance
            #increasing count
            count +=1
            #adding to total and respective color scores
            total_score += score
            if pad_color == 'blue':
                blue_score += score
            elif pad_color == 'pink':
                pink_score += score
            elif pad_color == 'grey':
                grey_score += score
    #calculating average score
    average_score = total_score / count if count > 0 else 0
    return total_score, average_score, blue_score, pink_score, grey_score

#main execution loop for processing images 1 to 10

#initalizing dictionaries and lists to hold scores and color wise data
camp_priority_score = {}
priority_ratio = {}
total_list_color_wise = []
#iterating through images 1 to 10
for x in range(1,11):
    image_path = f"C:\\Users\\yajat\\Code\\UAS-task2\\task_images\\{x}.png"
    
    #reading the original image and performing pre-processing
    original_image = cv2.imread(image_path)
    color_graded_image, ocean_mask, land_mask, protection_mask = pp.color_grade(image_path)

    #finding contours and separating casualties and rescue pads
    detected_objects = pp.find_contours(original_image, protection_mask)
    casualties, rescue_pads = pp.separate_casualties_and_pads(detected_objects)
    #drawing shapes and labels on the color graded image for visualization
    final_image = pp.draw_shapes_on_image(color_graded_image, detected_objects)

    #calculating priority and assignments
    pad_assignments, pad_casualty_colors = calculate_priority(casualties, rescue_pads)

    #preparing color wise casualty data for retrieval
    list_color_wise = [
        [item[:2] for item in pad_casualty_colors.get('blue', [])],
        [item[:2] for item in pad_casualty_colors.get('pink', [])],
        [item[:2] for item in pad_casualty_colors.get('grey', [])]
    ]
    print(f"image{x}.png :", list_color_wise)
    total_list_color_wise.append(list_color_wise)
    #drawing assignment arrows on the final image
    final_image_with_arrows = draw_assignment_arrows(final_image, pad_assignments)

    #can use if based on simple scoring system
    #total_score1, average_score1, blue_score1, pink_score1, grey_score1 = scores_sum_simple(pad_casualty_colors)

    #using the scoring system defined in the task
    total_score2, average_score2, blue_score2, pink_score2, grey_score2 = scores_sum_next(pad_casualty_colors)
    camp_priority_score[f"image_{x}.png"] = [blue_score2, pink_score2, grey_score2, total_score2]
    priority_ratio[f"image_{x}.png"] = average_score2

    # masking image, useful for debugging
    # cv2.imwrite(f"C:\\Users\\yajat\\Code\\UAS-task2\\output\\debug_protection_mask{x}.png", protection_mask)

    #saving the color graded image and final output
    cv2.imwrite(f"C:\\Users\\yajat\\Code\\UAS-task2\\color_graded_output\\color_graded_image{x}.png", color_graded_image)
    cv2.imwrite(f"C:\\Users\\yajat\\Code\\UAS-task2\\final_output\\final_output{x}.png", final_image_with_arrows)

    #print_assignments(assignments, casualties, rescue_pads)

#sorting images based on priority ratio for final output
priority_ratio_list = sorted(priority_ratio.items(), key=lambda item: -item[1])

#printing the priority scores for each image
for image_name, scores in camp_priority_score.items():
    print(f"{image_name}: Blue Priority: {scores[0]:.2f}, Pink Priority: {scores[1]:.2f}, Grey Priority: {scores[2]:.2f}, Total Priority: {scores[3]:.2f}")

#storing the order of images based on priority ratio
image_order = []
for img, score in priority_ratio_list:
    print(f"{img}: Priority Ratio {score:.2f}")
    image_order.append(img)
print(f"Order of priority of images based on priority ratio: {image_order}")

#defining various retrieval functions for use in other modules
#returns the casualty and pad data for a given image number
def retrieve_casualty_and_pad_data(image_number):
    return total_list_color_wise[image_number - 1]

#returns the priority scores for a given image number
def retrieve_data_of_image(image_number):
    image_tag = str(image_number)
    image_name = "image_" + image_tag + ".png"
    color_based_priority = []
    color_based_priority.append(round(float(camp_priority_score[image_name][0]),2))
    color_based_priority.append(round(float(camp_priority_score[image_name][1]),2))
    color_based_priority.append(round(float(camp_priority_score[image_name][2]),2))
    priority_ratio_value = round(float(priority_ratio[image_name]),2)
    return color_based_priority, priority_ratio_value

#returns the order of images based on priority ratio
def retrieve_image_order():
    return image_order

