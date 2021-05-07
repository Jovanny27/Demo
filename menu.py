#Menú para funciones
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
i = 0

while i != 4:
    print("Elige el filtro que desees usar")
    print("Opción 1. Filtro de borrego")
    print("Opción 2. Bordes")
    print("Opción 3. Filtro de máscara")
    print("Opción 4. Salir")
    i = int(input("Opción elegida: " ))
    if i == 1:
        print("Seleccionaste la opción 1")
        import cv2
        import imutils

        cap=cv2.VideoCapture(0)
        #imagenes a incrustar en el video
        #image = cv2.imread('gato.png', cv2.IMREAD_UNCHANGED)
        #image = cv2.imread('sombrerokul.png', cv2.IMREAD_UNCHANGED)
        image = cv2.imread('borre.png', cv2.IMREAD_UNCHANGED)
        #image = cv2.imread('A.jpg', cv2.IMREAD_UNCHANGED)
        #print('image.shape = ' , image.shape)
        #cv2.imshow('image', image[:,:, 3])

        #clasificador
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if ret == False: break
    
            #Deteccion de los rostros presentes en frame
            faces=faceClassif.detectMultiScale(frame, 1.3, 5)
    
            for(x, y, w, h) in faces:
            
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0 , 255, 0), 2)
        
                #ajuste al rostro
                resized_image = imutils.resize(image, width = w)
                filas_image = resized_image.shape[0]
                col_image = w
                #hacer que el filtro no se pierda
                dif=0
        
                #ajustar la imagen hacia la cara bajandola un poco
                porcion_alto = filas_image // 2
        
                # limites
                if y - filas_image + porcion_alto >=0:
                   n_frame = frame [y - filas_image + porcion_alto: y + porcion_alto, x: x + w]
           
                else:     
                   dif=abs(y - filas_image + porcion_alto)
                   n_frame = frame [0: y + porcion_alto, x: x + w]
               
                mask = resized_image[:,:,3]
                #mask = resized_image[:,:,2]
                mask_inv = cv2.bitwise_not(mask)
           
                bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
                bg_black = bg_black[dif:,:,0:3]
                bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:,:])
           
                #se suman las dos para que se juten y se obtenga el resultado final
                result = cv2.add(bg_black, bg_frame)
                if y - filas_image + porcion_alto >= 0:
                   frame [y - filas_image + porcion_alto: y + porcion_alto, x: x + w] = result
                else:
                    frame [0: y + porcion_alto, x: x + w] = result
                   #cv2.imshow('result',result)
                   #cv2.imshow('bg_frame',bg_frame)
           
           
            cv2.imshow('Frame',frame)
    
            k = cv2.waitKey(10) 
            if k==27: 
                break

        cap.release()
        cv2.destroyAllWindows()
    elif i == 2:
        print("Seleccionaste la opción 2")
        import cv2
        cap = cv2.VideoCapture(0) # 0 local camera
        con = 0
        while cap.isOpened():
            #BGR image feed from camera
            ret, img = cap.read()
            #BGR to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img_gray,10,20,apertureSize = 3)
  
  
            cv2.imshow("Edges", edges)
  
            k = cv2.waitKey(10)
            if k==27:
                break

        cap.release()
        cv2.destroyAllWindows()
    elif i == 3:
        cascPath = os.path.dirname(
            cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

        faceCascade = cv2.CascadeClassifier(cascPath)
        model = load_model("mask_recog1.h5")

        video_capture = cv2.VideoCapture(0)

        while True:
        # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(60, 60),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

            faces_list=[]
            preds=[]


            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h,x:x+w]
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_frame = cv2.resize(face_frame, (224, 224))
                face_frame = img_to_array(face_frame)
                face_frame = np.expand_dims(face_frame, axis=0)
                face_frame =  preprocess_input(face_frame)
                faces_list.append(face_frame)
                if len(faces_list)>0:
                    preds = model.predict(faces_list)
                for pred in preds:
                    (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (x, y- 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()

    elif i == 4:
        break
