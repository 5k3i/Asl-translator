import os
import cv2

path = "self-asl-dataset"
alphabet = ["a","b","c"]



# Cam stuff

tracker = 0
img_list = []

## =============== MY CODE ===================
def write_file_directory(alphabet: list[str], path: str):
    
    # for letter in alphabet:
    cap = cv2.VideoCapture(0)

def write_file_directory(alphabet, img_list, write_path):
    for letter in alphabet:
        for n in range(5):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(os.path.join(write_path, letter, letter+"_"+n+".jpg"))
                # cv2.imwrite(f"{write_path}/{letter}/{letter}_n.jpg", img_list[n-1])
                cv2.imwrite(os.path.join(write_path, letter, letter+"_"+n+".jpg"))


## =============== MY CODE ====================
    
# Function
def write_file_directory(img_list,write_path):
    alphabet = ["A","B","C"]

    for letter in alphabet:
        for i in range(5):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(f"{write_path}/{letter}/{letter}{i}.jpg", img_list[i-1])
                print(f"{write_path}/{letter}/{letter}{i}.jpg")



cam_capture = cv2.VideoCapture(0)
if not cam_capture.isOpened():
    print("Error opening video stream or file")
        

while True :
    ret, frame = cam_capture.read()
    if not ret:
        break
    if tracker >= 15 :
        break


    write_file_directory(alphabet, "C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        tracker += 1
        img_list.append(frame)
        if tracker % 5 == 0 :
            write_file_directory(img_list,"C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
        
    

    cv2.imshow("hi",frame)

cam_capture.release()
cv2.destroyAllWindows()
print(img_list)

#write_file_directory(img_list,"C:/Users/TechLab/Documents/Asl-Translator/Test-Folder")
     
