# Integrantes
José Adrián Porres Brugué y
Diego Sebastián Ortiz Sanchez
# proyecto3_facerecognition
https://docs.google.com/document/d/1nRzpX7zClwdnlMmgEE0AKf-US16jJ1F1Xc-DtRgZGf4/edit?usp=sharing
# Construccion del rtree:
Para la construcción del rtree tuvimos que tener acceso a la colección de imágenes a utilizar. Una vez descargadas y teniendo acceso a las imágenes, podemos observar que cada persona tiene su carpeta y dentro de esa misma tienen 1 o más imágenes. El algoritmo consiste en guardar las direcciones de los folders en un array e iterar en cada uno. Al tener uno de estos folderes, de la misma forma, obtenemos otro array de direcciones pero esta vez ya teniendo las imágenes e iteramos en él. Al iterar en cada imagen, la cargamos con la función load de la librería de  face_recognition para luego obtener el vector característico de la misma. Esta función de encoding, si es que la imagen tiene varias caras, trae una lista de vectores característicos de tamaño 128 para cada cara. Luego iteramos en estas, conseguimos sus coordenadas y finalmente lo insertamos en el rtree el cual se guardará como 2 archivos .idx y .dat. Esto se repetirá hasta completar el rtree con n imagenest.

Para la toma de datos del KNN-Rtree se tuvo que crear varios rtrees con el fin de tener diferentes tamaños y poder observar la decadencia de la eficacia conforme vamos aumentando la cantidad de imágenes.

En KNN-Secuencial iteramos en la carpeta lfw dentro de preprocess y ahí en cada una de las carpetas con las respectivas imágenes de las personas.

Truncado a 2 decimales:

k = 8

![Screenshot](capturaF.png)
