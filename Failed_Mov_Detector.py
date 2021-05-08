#Codigo proporcionado por la pagina : https://programarfacil.com/blog/vision-artificial/deteccion-de-movimiento-con-opencv-python/
import numpy as np
import cv2
import time

# Cargamos el vídeo
camara = cv2.VideoCapture("detector-movimiento-opencv.mp4")

# Inicializamos el primer frame a vacío.
# Nos servirá para obtener el fondo
fondo = None

# Recorremos todos los frames
while True:
	# Obtenemos el frame
	(grabbed, frame) = camara.read()

	# Si hemos llegado al final del vídeo salimos
	if not grabbed:
		break

	# Convertimos a escala de grises
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Aplicamos suavizado para eliminar ruido
	gris = cv2.GaussianBlur(gris, (21, 21), 0)

	# Si todavía no hemos obtenido el fondo, lo obtenemos
	# Será el primer frame que obtengamos
	if fondo is None:
		fondo = gris
		continue

	# Calculo de la diferencia entre el fondo y el frame actual
	resta = cv2.absdiff(fondo, gris)

	# Aplicamos un umbral
	umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)[1]

	# Dilatamos el umbral para tapar agujeros
	umbral = cv2.dilate(umbral, None, iterations=2)

	# Copiamos el umbral para detectar los contornos
	contornosimg = umbral.copy()

	# Buscamos contorno en la imagen
	im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Recorremos todos los contornos encontrados
	for c in contornos:
		# Eliminamos los contornos más pequeños
		if cv2.contourArea(c) < 500:
			continue

		# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
		(x, y, w, h) = cv2.boundingRect(c)
		# Dibujamos el rectángulo del bounds
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Mostramos las imágenes de la cámara, el umbral y la resta
	cv2.imshow("Camara", frame)
	cv2.imshow("Umbral", umbral)
	cv2.imshow("Resta", resta)
	cv2.imshow("Contorno", contornosimg)

	# Capturamos una tecla para salir
	key = cv2.waitKey(1) & 0xFF

	# Tiempo de espera para que se vea bien
	time.sleep(0.015)

	# Si ha pulsado la letra s, salimos
	if key == ord("s"):
		break

# Liberamos la cámara y cerramos todas las ventanas
camara.release()
cv2.destroyAllWindows()



















#intento de combinar el algorintmo con el de camera_python.py
import numpy as np 
import cv2
import time
import argparse


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Camera visualization')

    ### Positional arguments
    parser.add_argument('-i', '--cameraSource', default=0, help="Introduce number or camera path, default is 0 (default cam)")

    
    args = vars(parser.parse_args())


    cap = cv2.VideoCapture(args["cameraSource"]) #0 local o primary camera

    while cap.isOpened():
        
        #BGR image feed from camera
        success,img = cap.read()
        
        fondo = None
        while True:
            (grabbed, frame) = cap.read()
            if not grabbed:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if fondo is None:
                fondo = gray
                continue
            resta = cv2.absdiff(fondo, gray)
            umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)
            umbral = cv2.dilate(umbral, None, iterations=2)
            contornosimg = umbral.copy()
            im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in contornos:
                if cv2.contourArea(c) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            cv2.imshow("Camara", frame)
            cv2.imshow("Umbral", umbral)
            cv2.imshow("Resta", resta)
            cv2.imshow("Contorno", contornosimg)
                
            key = cv2.waitKey(1) & 0xFF
                
            time.sleep(0.015)
            if key == ord("s"):
                break

        
        if not success:
            break
        if img is None:
            break

        
        cv2.imshow("Output", img)

        k = cv2.waitKey(10)
        if k==27:
            break
    
            

    cap.release()
    cv2.destroyAllWindows()

    print('Script took %f seconds.' % (time.time() - script_start_time))

import numpy as np
import cv2
import time
import argparse

if __name__ == '__main__':
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description='Camera visualization')
    parser.add_argument('-i', '--cameraSource', default=0, help="Introduce number or camera path, default is 0 (default cam)")
    args = vars(parser.parse_args())
    cap = cv2.VideoCapture(args["cameraSource"])
    fondo = None
    while cap.isOpened():
        (grabbed,frame) = cap.read()
        if not grabbed:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if fondo is None:
            fondo = gray
            continue
        
        resta = cv2.absdiff(fondo, gray)
        umbral = cv2.threshold(resta, 40, 255, cv2.THRESH_BINARY)
        umbral = cv2.dilate(umbral, None, iterations=2)
        contornosimg = umbral.copy()
        im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            if cv2.contourArea(c) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Output", img)
        cv2.imshow("Camara", frame)
        cv2.imshow("Umbral", umbral)
        cv2.imshow("Resta", resta)
        cv2.imshow("Contorno", contornosimg)

        
        key = cv2.waitKey(10) & 0xFF
        time.sleep(0.015)
        if key == ord("s"):
            break
    camara.release()
    cv2.destroyAllWindows()
    print('Script took %f seconds.' % (time.time() - script_start_time))






#otro algoritmo para la misma función
# Importación de librerías
import numpy as np
import cv2

# Capturamos el vídeo
cap = cv2.VideoCapture('detector-movimiento-opencv.mp4')

# Llamada al método
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)

# Deshabilitamos OpenCL, si no hacemos esto no funciona
cv2.ocl.setUseOpenCL(False)

while(1):
	# Leemos el siguiente frame
	ret, frame = cap.read()

	# Si hemos llegado al final del vídeo salimos
	if not ret:
		break

	# Aplicamos el algoritmo
	fgmask = fgbg.apply(frame)

	# Copiamos el umbral para detectar los contornos
	contornosimg = fgmask.copy()

	# Buscamos contorno en la imagen
	im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Recorremos todos los contornos encontrados
	for c in contornos:
		# Eliminamos los contornos más pequeños
		if cv2.contourArea(c) < 500:
			continue

		# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
		(x, y, w, h) = cv2.boundingRect(c)
		# Dibujamos el rectángulo del bounds
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Mostramos las capturas
	cv2.imshow('Camara',frame)
	cv2.imshow('Umbral',fgmask)
	cv2.imshow('Contornos',contornosimg)

	# Sentencias para salir, pulsa 's' y sale
	k = cv2.waitKey(30) & 0xff
	if k == ord("s"):
		break

# Liberamos la cámara y cerramos todas las ventanas
cap.release()
cv2.destroyAllWindows()






#algoritmo basado en el expuesto por el video:https://www.youtube.com/watch?v=kcmJQzu_q6M&t=386s y comparado con el de camera_python.py
import cv2
import numpy as np

video = cv2.VideoCapture('Video2.mp4')
i=0
while True:
    ret,frame = video.read()
    if ret==False:
        break
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if i==20:
        bgGray = gray
    if i>20:
        dif =cv2.absdiff(gray, bgGray)
        _, th =cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
        _, cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("th", th)
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 9000:
                (x,y,w,h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0), 2)
                 
    cv2.imshow("frame", frame)

    i=i+1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()        
cv2.destroyAllWindows()
