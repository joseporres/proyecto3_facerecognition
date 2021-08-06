import face_recognition
from rtree import index 
import os

n = 100#trabajaremos con 100 imagenes para ver el funcionamiento de nuestros algoritmos

def preprocess():    
    path = "lfw"
    p = index.Property()
    p.dimension = 128
    p.buffering_capacity = 10
    Rtree = index.Index('Rtree100', properties=p)

    listNames = os.listdir(path)

    id = 0
    for name in listNames:
        imgList = os.listdir(path + "/" + name)
        for img in imgList: 
            #print(path + "/" + name + "/" + img)
            
            # obtenemos las caras
            imgPro = face_recognition.load_image_file(path + "/" + name + "/" + img)

            # vectores de las caras en la imagen
            facesEncoding = face_recognition.face_encodings(imgPro)

            # llenar el Rtree
            for faceEncoding in facesEncoding:
                listAux = list(faceEncoding)
                for cord in faceEncoding:
                    listAux.append(cord)
                Rtree.insert(id, listAux, ("preprocess/" + path + "/" + name + "/" + img))
                id += 1

            if id == n:
                return Rtree
    Rtree.close()

    return Rtree      



# preprocess()