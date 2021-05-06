#Menú para funciones

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

        cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        """
           Primero importanmos las librerias que se utilizarán en el prgrama.
           Cabe recalcar que también se debe de descargar un modelo pre-hecho que nos va a servir para poder identficar entre
           la cara con cubrebocas y sin cubrebocas el cual lleva como nombre "mask_recog1.h5"
        """
        import cv2
        import os
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        import numpy as np
        """
            El primer paso es encontrar la ruta al archivo "haarcascade_frontalface_alt2.xml".
            Hacemos esto usando el módulo os del lenguaje de Python.
        """
        cascPath = os.path.dirname(
            cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        """
            El siguiente paso es cargar nuestro clasificador. La ruta al archivo xml anterior va
            como argumento del método CascadeClassifer() de OpenCV. También cargamos el modelo
            previamente descargado el cual debe de estar en la misma carpeta donde se corre este programa.
        """
        faceCascade = cv2.CascadeClassifier(cascPath)
        model = load_model("mask_recog1.h5")
        """
            Después de cargar el clasificador, abrimos la cámara usando
            este simple código OpenCV de una sola línea.
        """
        video_capture = cv2.VideoCapture(0)
        """
            A continuación, necesitamos obtener los fotogramas de la transmisión de la cámara,
            lo hacemos usando la función read(). Lo usamos en bucle infinito para obtener todos
            los fotogramas hasta el momento en que queremos cerrar la transmisión.
        """
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
        """
            Para que este clasificador específico funcione, necesitamos convertir el marco
            en escala de grises.
        """            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """
            El faceCascade tiene un método detectMultiScale(), que recibe un marco(imagen) como
            argumento y ejecuta la cascada del clasificador sobre la image. El término MultiScale
            indica que el algoritmo mira subregiones de la imagen en múltiples escalas, para
            detectar caras de diferentes tamaños.
        """
            faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        """
            A continuación, definimos algunas listas. Face_list contiene todas las caras que son detectadas
            por el modelo FaseCascade y la lista de preds se usa para almacenar las predicciones hechas por
            el modelo detector de máscara.
        """
            faces_list=[]
            preds=[]
        """
            Dado que la variable de caras contiene las coordenadas de la esquina superior izquierda, la altura
            y el ancho del rectángulo que abarca las caras, podemos usar eso para obtener un marco de la cara
            y luego preprocesar ese marco para que pueda introducirse en el modelo para la predicción.
        """
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
        """
            Depués de obtener las predicciones, dibujamos un rectángulo sobre la cara y colocamos
            una etiqueta de acuerdo con las predicciones.
        """
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        """
            El cv2.rectangke() dibuja rectángulos sobre las imagenes y necesita conocer las coordenadas
            de píxeles de la esquina superior izquierda e inferior derecha. Las coordenadas indican la
            fila y columna de píxeles de la imagen.
        """
            cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
        """
            A continuación, solo mostramos el cuadro resultante y también establecemos una forma de salir
            de este bucle infinito y cerrar la transmición de video. Pulsando la tecla "esc", podemos salir
            del script.
        """
            # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        video_capture.release()
        cv2.destroyAllWindows()
    elif i == 4:
        break
