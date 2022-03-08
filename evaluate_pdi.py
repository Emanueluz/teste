import cv2
import torch
import os
from src.data.basic_transforms import get_image_transforms
from csv import reader
from sklearn.metrics import classification_report
from tqdm import tqdm
from matplotlib import pyplot as plot
import numpy as np
#abrir o arquivo 
def abrir(arq_csv):
    with open(arq_csv, 'r') as ficheiro:
        reader = csv.reader(ficheiro)
        reader= list(reader)
        
    return reader

def abrir_csv(arq_csv):

    with open(arq_csv, 'r') as csv_file:
        csv_reader = reader(csv_file,delimiter=';')
        list_of_rows = list(csv_reader)

    return list_of_rows[1:]


def f_gabarito(lista):
    aux=[]
    c=0
    for i in lista:
        for j in range(1,7):
            if i[j]!="0":
                c=1         
        aux.append([i[0],c])
        c=0
    return aux


def caminho(arq):
    lista_de_arq=os.listdir(arq)
    return lista_de_arq

def pdi(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    eroded = cv2.erode(gray, kernel)
    dilated = cv2.dilate(eroded, kernel)

    ret, labels = cv2.connectedComponents(gray, connectivity=4)
    unique, counts = np.unique(labels, return_counts=True)

    if len(unique)>1:
        return 1
    else:
        return 0

def main(imagem_png,csv,arq_saida):
    lista_png=caminho(imagem_png)
    lis_csv=abrir_csv(csv)


    resultados = []
    gabarito = []
    
    for imagem in range(len(lis_csv)):
        if lis_csv[imagem][0]+".png" in lista_png:
            print(imagem_png+"\\%s"%lis_csv[imagem][0]+".png")
            img =cv2.imread(imagem_png+"\\%s"%lis_csv[imagem][0]+".png")
            resultados.append(pdi(img))
            gabarito.append(int(lis_csv[imagem][-1]))
        

    with open(arq_saida,"w") as saida:
        
        saida.write(classification_report(gabarito,resultados))
    return

if __name__ == '__main__':
    imagem_png='E:\\hd1\\rsna-hemorrhage\\genesis-brain-hemorrhage-main\\data\\rsna\\png\\Brain'
    arq_saida="saida_pdi.txt"
    csv="E:\\hd1\\rsna-hemorrhage\\genesis-brain-hemorrhage-main\\data\\rsna\\rsna.csv"
    main(imagem_png,csv,arq_saida)