# Evidencia Equipo 1
Integrantes: 
- Jovanny Rafael Ramirez | A01639287
- Agustín Gutiérrez Talavera | A01745019
- Ana Luisa Montero | A01638796
- Diana Rojas | A01610999
- Ximena Aquino Pérez | A01639678
- Samir Ortiz | A01639922
- Diego Kury | A00227097
# Objetivo del proyecto
Durante esta Semana Tec aprendimos profundizamos más acerca del uso de Python, más específicamente en la descarga y uso de distintas librerías, principalmente la de OpenVC, la cual es una herramienta muy útil para utilizar la visión de la cámara mediante Python. El objetivo de nuestro proyecto es mostrar algunas de las posibles aplicaciones que se le pueden dar a la visión de cámara en Python.
## Instalar dependencias necesarias:
- Python3
- OpenCV
- TensorFlow
- pip 21.1.1
- Brew
- Mtcnn
- imutils
# Descripción general del proyecto 
Nuestro proyecto cuenta principalmente de tres diferentes aplicaciones dentro de las cuales se utiliza el video y de un menú para poder elegir la aplicación que se desee. Las tres aplicaciones parten del uso de la biblioteca OpenVC pero utilizan distintas librerías dependiendo según el propósito de la aplicación.
# MENÚ
Al correr el código, lo primero que aparece es el menú, el cual da a elegir cuál es la aplicación que se desea utilizar.
# App de filtro
Esta aplicación consiste en colocar una imagen en la frente de las personas que la cámara detecte. Algunas de las consideraciones de este filtro es que su tamaño se ajusta dependiendo de la distancia entre la persona y la cámara, y que se encuentra programado para solo aparecer en la frente. 
![filtro python](https://user-images.githubusercontent.com/83722304/117389921-cbb54b80-aeb2-11eb-9043-48723f738805.PNG)
Haciendo uso de la librería cv2, llamamos a cv2.VideoCapture para iniciar el proceso de la cámara. Asímismo, se hicieron varios ajustes en el código para confirmar que el filtro se mantenga en el marco y donde debe de estar posicionado respecto a la cara.
![img1](https://user-images.githubusercontent.com/83785021/117402417-379bb200-aebb-11eb-9aad-d3719eb9d13a.png)


# App de distorción de color
Consiste en distorsionar el color del streaming de la cámara por un color negro de fondo, definiendo los contornos de las figuras con lineas blancas y rellenando los curpos de pequeñas figuras blancas irregulares.
![edges python](https://user-images.githubusercontent.com/83722304/117390981-b3dec700-aeb4-11eb-8bbb-cf40a11d2ddf.PNG)
# Especificaciones del uso del filtro de distorción 
- Se recomienda que el lugar donde se coloque la cámara tenga una buena iluminación para que el filtro pueda alcanzar a definir bien todos los objetos.
# App detectora de cubrebocas
Consiste básicamente en detectar cuando una persona está utilizando cubrebocas o no, marcando en un rectángulo color verde cuando la persona lleva el cubrebocas puesto y en un rectángulo rojo cuando no.
![image](https://user-images.githubusercontent.com/83722304/117391735-30be7080-aeb6-11eb-8e37-0e78777a9a99.png)
![image](https://user-images.githubusercontent.com/83722304/117391769-40d65000-aeb6-11eb-84fb-40e9709b4505.png)
# Especifcaciones del uso del detector del cubrebocas
- Previo a la ejecución del programa, se debe cargar un documento que funcione como "clasificador", es decir, que especifique el caso en el que será positivo y el caso en el que será negativo.
- Para que el clasificador funcione, se debe convertir el marco en escala de grises.
# Autores
- Jovanny Rafael Ramirez | A01639287 Contribuyente
- Agustín Gutiérrez Talavera | A01745019 Contribuyente 
- Ana Luisa Montero | A01638796 Contribuyente
- Diana Rojas | A01610999 Contribuyente
- Ximena Aquino Pérez | A01639678 Contribuyente
- Samir Ortiz | A01639922 Contribuyente
- Diego Kury | A00227097 Contribuyente
- OMES (Youtube). https://www.youtube.com/watch?v=R6wgJ6epakU - Documentación
- 
