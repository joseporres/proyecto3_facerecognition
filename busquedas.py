import face_recognition
from rtree import index 
import os
import heapq
import json
import linecache

def knnSequential(k, Q, n):  
    cont = 0  
    ruta = []
    caras = []

    for pathF in os.listdir("preprocess/lfw"):
        auxPath = "preprocess/lfw/" + pathF
        images = os.listdir(auxPath)

        for nameF in images: 
            auxPath2 = auxPath + "/" + nameF
            for it in face_recognition.face_encodings(face_recognition.load_image_file(auxPath2)):
                if cont == n:
                    dist = face_recognition.face_distance(caras, Q)
                    auxArr = []
                    for i in range(0, cont, 1):
                        auxArr.append((dist[i], ruta[i]))
                    heapq.heapify(auxArr)    
                    return heapq.nsmallest(k, auxArr)

                ruta.append(auxPath2)
                caras.append(it)        
                cont += 1


    dist = face_recognition.face_distance(caras, Q)
    auxArr = [] 

    for i in range(0, cont, 1):
        auxArr.append((dist[i], ruta[i]))
    heapq.heapify(auxArr)    
    return heapq.nsmallest(k, auxArr)


def rangeSearch(r, Q, n):
    p = index.Property()
    p.dimension = 128
    p.buffering_capacity = 3
    Rtree = index.Rtree("preprocess/Rtree"+ str(n), properties=p)  
    lim = []
    lim.extend( [ i - r for i in Q ])
    lim.extend( [ i + r for i in Q ])
    return [x.object for x in Rtree.intersection(lim, objects=True)]

def knnRtree(k, Q, n):
  p = index.Property()
  p.dimension = 128 
  p.buffering_capacity = 10 
 
  Rtree = index.Rtree("preprocess/Rtree"+ str(n), properties=p)  
  listQ = list(Q)
  for i in Q:
    listQ.append(i)

  return list(Rtree.nearest(coordinates=listQ, num_results=k, objects='raw'))

# print(rangeSearch(0.2
# , face_recognition.face_encodings(face_recognition.load_image_file('fotos_prueba/unknown.jpg'))[0]
# , 100))

# print(knnRtree(5
# , face_recognition.face_encodings(face_recognition.load_image_file('fotos_prueba/unknown.jpg'))[0]
# , 100))

# print(knnSequential(5
# , face_recognition.face_encodings(face_recognition.load_image_file('fotos_prueba/unknown.jpg'))[0]
# , 100))

