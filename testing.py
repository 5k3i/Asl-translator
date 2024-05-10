import os
import cv2

import sys
# TODO: TASKS
# Taking Photos [-]
# Train model again [ ]
# Train with different skin tones [ ]
# Write letters to screen [ ]
# Mimic keyboard input (spacebar) to form a sentence [?]
# FUTURE: Text to Speech [ ]
# 

path = "self-asl-dataset"
alphabet = ["a","b","c","d","e"]



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

    for i in range(10):
        cv2.imwrite(f"{write_path}/{alphabet[alphabet_iteration]}/{alphabet[alphabet_iteration]}{i+10}.jpg", img_list[i-1])
        print(f"{write_path}/{alphabet[alphabet_iteration]}/{alphabet[alphabet_iteration]}{i}.jpg")

# WHAT WAS I GONNA DO WITH THISS
if len(sys.argv) > 2:
    raise NotImplementedError("arguments not validated:")

cam_capture = cv2.VideoCapture(0)
if not cam_capture.isOpened():
    print("Error opening video stream or file")
        
#for letter in alphabet:
    #os.mkdir(letter)

while True :
    ret, frame = cam_capture.read()
    if not ret:
        break
    if tracker >= 110 :
        break


    #write_file_directory(alphabet, "C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        tracker += 1
        img_list.append(frame)
        if tracker % 10 == 0 :
            write_file_directory(img_list,"C:/Users/TechLab/Documents/Asl-Translator/short-test",alphabet_iteration)
            alphabet_iteration += 1
            img_list = []
        
    

    cv2.imshow("hi",frame)

cam_capture.release()
cv2.destroyAllWindows()



#write_file_directory(img_list,"C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
     
