# Deep Learning Based Attendance System (DBAS)

An attendance system implemented using PCN (Face Detection), FaceNet (Face Recognition) and Flask 

### Libraries used:
* pytorch  
* torchvision  
* cudatoolkit  
* opencv  
* flask  
* flask_wtf  
* flask_pymongo  

### Database Used:
MongoDB Local database

### IP Webcam app is used to stream mobile phone video to opencv
https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_IN

Start a server from the IP Webcam app, access the IP provided by the app on the PC (phone and PC should be on the same network)

Then pass `http://<ip-from-IP-WebCam>/shot.jpg` to faceRec_flask file.

In the project root, run `flask run` to start the flask server

### Codes used:
* PCN: 
https://github.com/siriusdemon/pytorch-PCN

* Facenet: 
https://github.com/parvatijay2901/FaceNet_FR
https://github.com/timesler/facenet-pytorch