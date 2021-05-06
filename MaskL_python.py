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
    k = cv2.waitKey(1)
    if k == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
