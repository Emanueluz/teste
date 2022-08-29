import cv2
import torch
import os
from csv import *
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



def escrever(nome_do_arq,hemorragia):

    with open(nome_do_arq+".txt", 'w') as aa:
        for i in hemorragia:        
            aa.write(str(i)+'\n')
    return






def listagem_hemo(lista):
    aux=[]
    freq=[["epidural",0],["intraparenchymal",0],["intraventricular",0],["subarachnoid",0],["subdural",0],["any",0]]
    hemo=[[],[],[],[],[],[]]
    c=0
    
    for i in tqdm(lista):
        for j in range(1,7):
            if i[j]=="1":
                hemo[j-1].append(i)
                c=1         
        aux.append([i[0],c])
        c=0
    
    return hemo


def f_gabarito(lista):
    aux=[]
    freq=[["epidural",0],["intraparenchymal",0],["intraventricular",0],["subarachnoid",0],["subdural",0],["any",0]]
    c=0
    
    for i in tqdm(lista):
        for j in range(1,7):
            if i[j]!="0":
                freq[j-1][1]=freq[j-1][1]+1
                c=1         
        aux.append([i[0],c])
        c=0
    print("len = ",len(lista),freq)
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
    freq=["epidural","intraparenchymal","intraventricular","subarachnoid","subdural","any"]
    a=listagem_hemo(lis_csv)
     

    return

if __name__ == '__main__':
    imagem_png='E:\\hd1\\rsna-hemorrhage\\genesis-brain-hemorrhage-main\\data\\rsna\\png\\Brain'
    arq_saida="saida_pdi.txt"
    csv="E:\\hd1\\rsna-hemorrhage\\genesis-brain-hemorrhage-main\\data\\rsna\\all_data.csv"
    main(imagem_png,csv,arq_saida)
