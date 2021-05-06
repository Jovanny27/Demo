import cv2
import imutils

cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
#imagenes a incrustar en el video
#image = cv2.imread('gato.png', cv2.IMREAD_UNCHANGED)
#image = cv2.imread('sombrerokul.png', cv2.IMREAD_UNCHANGED)
image = cv2.imread('borre.png', cv2.IMREAD_UNCHANGED) # se seleciiona la imagen con formato png
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
# Este codigo fue extraido de el video de YouTube con direccion https://www.youtube.com/watch?v=R6wgJ6epakU
#  Del canal OMES
