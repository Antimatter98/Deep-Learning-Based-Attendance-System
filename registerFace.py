import cv2
import os

def registerFace(reg_name):
    cam = cv2.VideoCapture(0)
    cnt = 0

    print("Student Registration/n")
    print("Registration for student: ", reg_name)
    # reg_name = input("Enter student name: ")

    # roll_no = input("Enter student Roll No: ")

    # mail = input("Enter student email: ")

    # password = input("Enter password: ")

    img_name = ''
    path = "D:\\Facenet-PCN-test\\student_data\\Test\\"+reg_name
    os.mkdir(path)
    cnt = 0
    cv2.namedWindow("Welcome to DBAS")
    fa=cv2.CascadeClassifier('faces.xml')

    while True:
        ret, frame = cam.read()
        cv2.imshow("Welcome to DBAS", frame)
        frame=cv2.flip(frame, 1)
        if not ret:
            break
        k = cv2.waitKey(1)

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=fa.detectMultiScale(gray, 1.3, 5)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            #img_name = reg_name+str(cnt)+".jpg"
            for (x,y,w,h) in faces:
                cropped=frame[y:y+h,x:x+w]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,cv2.LINE_AA)
                gr = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
                #pic_no=pic_no+1
                cv2.imwrite(path+'\\'+str(cnt)+'.jpg',gr)
                #cnt = cnt + 1
                cnt+=1
            cv2.imshow('frame',frame)
            #cv2.imwrite(os.path.join(path,img_name),frame)
            print("Image captured! "+str(cnt)+"/4")
            
            if cnt == 5:
                print("Registration successful")
                break
                
    cam.release()

    cv2.destroyAllWindows()

    return reg_name