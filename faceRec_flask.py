
from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import random

import requests
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear

from facenet_pytorch.models.mtcnn import MTCNN
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
# from facenet_pytorch import MTCNN, InceptionResnetV1,extract_face
from facenet_pytorch import extract_face
from PIL import Image,ImageDraw
import cv2
import os
import torch.nn as nn
from Web.imutils.video import WebcamVideoStream
from Web.imutils.video import FPS
import imutils

import pcn
from pcn.utils import rotate_point, crop_face
from utils.face import load_embeddings, add_new_user, save_embeddings, run_embeddings_knn
from utils.utils import static_vars
import argparse
import dlib
import time


def calcEmbedsRec(urlNew):
    
    #initialize identified names
    recognized_names = []
    
    print('Received url: ', urlNew)
    device = torch.device('cuda:0') 
    print('Running on device: {}'.format(device))
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device
    )
    #Function takes 2 vectors 'a' and 'b'
    #Returns the cosine similarity according to the definition of the dot product
    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    #cos_sim returns real numbers,where negative numbers have different interpretations.
    #So we use this function to return only positive values.
    def cos(a,b):
        minx = -1 
        maxx = 1
        return (cos_sim(a,b)- minx)/(maxx-minx)

    # Define Inception Resnet V1 module (GoogLe Net)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Define a dataset and data loader
    dataset = datasets.ImageFolder('student_data/Test')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    #Perfom MTCNN facial detection
    #Detects the face present in the image and prints the probablity of face detected in the image.
    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    # Calculate the 512 face embeddings
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).to(device)

    # Print distance matrix for classes.
    #The embeddings are plotted in space and cosine distace is measured.
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(0,len(names)):
        emb=embeddings[i].unsqueeze(0)
        # The cosine similarity between the embeddings is given by 'dist'.
        dist =cos(embeddings[0],emb)  
            
    dists = [[cos(e1,e2).item() for e2 in embeddings] for e1 in embeddings]
    # The print statement below is
    #Helpful for analysing the results and for determining the value of threshold.
    print(pd.DataFrame(dists, columns=names, index=names)) 


    i = 1
    # Haarcascade Classifier is used to detect faces through webcam. 
    #It is preffered over MTCNN as it is faster. Real time basic applications needs to be fast.
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #Takes 2 vectors 'a' and 'b' .
    #Returns the cosine similarity according to the definition of the dot product.
    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    #cos_sim returns real numbers,where negative numbers have different interpretations.
    #So we use this function to return only positive values.
    def cos(a,b):
        minx = -1 
        maxx = 1
        return (cos_sim(a,b)- minx)/(maxx-minx)

    #This is the function for doing face recognition.
    def verify(embedding, start_rec_time):
        for i,k in enumerate(embeddings):
            for j,l in enumerate(embedding):
                #Computing Cosine distance.
                dist =cos(k,l)
                
                #Chosen threshold is 0.85
                #Threshold is determined after seeing the table in the previous cell.
                if dist > 0.8:
                    #Name of the person identified is printed on the screen, as well as below the detecetd face (below the rectangular box).
                    text= names[i]

                    #textOnImg = text + " - Time Elapsed: " +  str(int(time.time() - start_rec_time)) + " s"
                    cv2.putText(img1, text, (boxes[j][0].astype(int),boxes[j][3].astype(int) + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
                    #cv2.putText(img1, textOnImg, (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
                    print(text)

                    #if text in names:
                    recognized_names.append(text)
                #else:
                textOnImg = "Time Elapsed: " +  str(int(time.time() - start_rec_time)) + " s"
                cv2.putText(img1, textOnImg,(20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)

    #Define Inception Resnet V1 module (GoogLe Net)                    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device,keep_all=True
    )

    #Camera is opened. Webcam video streaming starts.
    #vs = WebcamVideoStream(src=0).start()
    print("Camera on")
    cv2.namedWindow("Detected faces")

    options = {"CAP_PROP_FRAME_WIDTH":640, "CAP_PROP_FRAME_HEIGHT":480, "CAP_PROP_FPS ":30}
    output_params = {"-fourcc":"MJPG", "-fps": 30}
    writer = WriteGear(output_filename = 'Output.mp4', compression_mode = False, logging = True, **output_params)
    #stream = VideoGear(source=0, time_delay=1, logging=True, **options).start()
    
    #url = "http://192.168.43.223:8080/shot.jpg"
    url = urlNew

    #run face recognition for 1 minute
    start_face_rec = time.time()
    end_face_rec = time.time() + 60

    while (time.time() < end_face_rec):
        
        # frm = stream.read()
        # if frm is None:
        #     break

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

        img = cv2.imdecode(img_arr, -1)

        #im= vs.read()
        #Flip to act as a mirror

        im=cv2.flip(img,1) 
        
        #try:
        #The resize function of imutils maintains the aspect ratio 
        #It provides the keyword arguments width and heightso the image can be resized to the intended width/height 
        frame = imutils.resize(im, width=400)
        
        #Detecting faces using Haarcascade classifier.

        winlist = pcn.detect(frame)
        img1 = pcn.draw(frame, winlist)
        face = list(map(lambda win: crop_face(img1, win, 160), winlist))
        face = [f[0] for f in face]
        #cv2.imshow('Live Feed', img1)
        cnt = 1
        for f in face:
            #fc, u = crop_face(img, f)
            print('Printing Face no: ', cnt)
            cv2.imshow('Detected faces', f)
            cnt += 1

            #faces = classifier.detectMultiScale(face)
            path="./student_data/Pics/".format(i)
            img_name = "image_{}.jpg".format(i)  
            #The captured image is saved.
            cv2.imwrite(os.path.join(path,img_name),f)
            imgName="./student_data/Pics/image_{}.jpg".format(i)

            # Get cropped and prewhitened image tensor
            img=Image.open(imgName)
            i=i+1
            img_cropped = mtcnn(img)
            boxes,prob=mtcnn.detect(img)
            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            #print(boxes)
            #Rectangular boxes are drawn on faces present in the image.
            #The detected and cropped faces are then saved.
            if(boxes is not None):
                for i, box in enumerate(boxes):
                    #draw.rectangle(box.tolist())
                    extract_face(img, box, save_path='./student_data/Pics/Cropped_Face_{}.jpg'.format(i))
                img_draw.save('./student_data/Pics/Faces_Detected.jpg')
                ima=cv2.imread('./student_data/Pics/Faces_Detected.jpg')

                #Calculate embeddings of each cropped face.
            if(img_cropped is not None):
                img_embedding = resnet(img_cropped.cuda()).to(device)

                #Call function verify. 
                #Identify the person with the help of embeddings.
                cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
                verify(img_embedding, start_face_rec)
            #else:
                #textForImg = "Time Elapsed: " +  str(int(time.time() - start_face_rec)) + " s"
                #cv2.putText(frame, textForImg, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            #'Detecting..' window opens.
            #Rectangular boxes are drawn on detected faces.
            #The identified faces have their respective name below the box.
            cv2.imshow('Detecting...',img1)
            writer.write(img1) 
        
        
        if(not face):
            #cv2.imshow(f"Time Elapsed: ${str(int(time.time() - start_face_rec))}  s" ,frame)
            textForImg = "Time Elapsed: " +  str(int(time.time() - start_face_rec)) + " s"
            cv2.putText(img1, textForImg, (40, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
            #print("no face")
            cv2.imshow('Detecting...',img1)
        # except:
        #     #In case 'try' doesn't work, "Get the image embedding" text is printed on the screen.
        #     #Run first cell
        #     text="Get the image embeddings"
        #     print(text)
        #     break 
            
        key = cv2.waitKey(1)
        
        #13 is for 'Enter' key.
        #If 'Enter' key is pressed, all the windows are made to close forcefully.
        if key ==13:
            break    

    print("calculating a list of all recognized faces...")

    rec_names_dict = {i:recognized_names.count(i) for i in recognized_names}

    filtered_names = []
    for key in rec_names_dict:
        if rec_names_dict[key] > 30:
            filtered_names.append(key)

    print("Total Recognized names: ", rec_names_dict)

    print("Filtered names: ", filtered_names)

    cv2.destroyAllWindows() 
    writer.close()
    #vs.stop()
    #return {i:rec_names_dict[i] for i in filtered_names}
    return filtered_names