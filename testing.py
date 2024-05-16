import os
import cv2

import sys
import matplotlib.pyplot as plt
# TODO: TASKS
# Taking Photos [-]
# Train model again [ ]
# Train with different skin tones [ ]
# Write letters to screen [ ]
# Mimic keyboard input (spacebar) to form a sentence [?]
# FUTURE: Text to Speech [ ]
# 

path = "self-asl-dataset"

alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"] 


# Cam stuff

alphabet_iteration = 0
tracker = 0
img_list = []

## =============== MY CODE ===================
# def write_file_directory(alphabet: list[str], path: str):
    
#     # for letter in alphabet:
#     cap = cv2.VideoCapture(0)

# def write_file_directory(alphabet, img_list, write_path):
#     for letter in alphabet:
#         for n in range(5):
#             if cv2.waitkeu
#             cv2.imwrite(f"{write_path}/{letter}/{letter}_n.jpg", img_list[n-1])


## =============== MY CODE ==========
    
# Function
def write_file_directory(img_list,write_path,alphabet_iteration):
    alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"] 

    for i in range(25):
        cv2.imwrite(f"{write_path}/{alphabet[alphabet_iteration]}/{alphabet[alphabet_iteration]}{i}.jpg", img_list[i-1])
        print(f"{write_path}/{alphabet[alphabet_iteration]}/{alphabet[alphabet_iteration]}{i}.jpg")

# WHAT WAS I GONNA DO WITH THISS
if len(sys.argv) > 2:
    if sys.argv[2] == "--overwrite":
        os.rmdir("folder")

cam_capture = cv2.VideoCapture(0)
if not cam_capture.isOpened():
    print("Error opening video stream or file")
        
#for letter in alphabet:
    #os.mkdir(letter)

record = False
idx = 0
frames = 0
alphabet = "ABCDEFGHIJKLMNO"


while True:
    ret, frame = cam_capture.read()
    if not ret:
        break
    # if tracker >= 125 :
    # 
    #     break

    if record and tracker % 5 == 0: 
        if not os.path.exists("folder"):
            os.mkdir("folder")  
        if not os.path.exists(f"./folder/{alphabet[idx]}"):
            os.mkdir(f"./folder/{alphabet[idx]}")     
        cv2.imwrite(f"./folder/{alphabet[idx]}/{alphabet[idx]}_right_{frames}.jpg", frame)
        frames += 1
        print(f"Letter: {alphabet[idx]}")
        print(f"frame: {frames}")
        
        # Reshoot training data for K

        if frames == 300:
            idx += 1
            record = False
            frames = 0
            tracker = 0
    


    if record :
        tracker += 1
    #write_file_directory(alphabet, "C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        record = True
    
        img_list.append(frame)
        # print("Photo Taken!", tracker)
        # if tracker % 25 == 0 :
        #     write_file_directory(img_list,"C:/Users/TechLab/Documents/Asl-Translator/final-train-data",alphabet_iteration)
        #     alphabet_iteration += 1
        #     img_list = []
    

    cv2.imshow("hi",frame)


cam_capture.release()
cv2.destroyAllWindows()




#write_file_directory(img_list,"C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
     
