import cv2
 
import os
from csv import reader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from matplotlib import pyplot as plot
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
def lista_contem(lista, lista_png):
    pertence=[]
    for i in lista:
        j=0
        while (j< len(lista_png)):
            
            if i[0]+".png"==lista_png[j]:
                pertence.append(i)
                j=len(lista_png)
            j=j+1

    return pertence


def abrir(arq_csv):
    with open(arq_csv, 'r') as ficheiro:
        lista_d_caso=[]
        a=ficheiro.readlines()
        for i in a:
            b=i.replace("\n", "")
            b=b.replace(" ", "")
            lista_d_caso.append(b.split(";"))
        
    return lista_d_caso

def caminho(arq):
    lista_de_arq=os.listdir(arq)
    return lista_de_arq


def pdi(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    ret, labels = cv2.connectedComponents(gray, connectivity=4)
    unique, counts = np.unique(labels, return_counts=True)

    if len(unique)>1:
        return 1
    else:
        return 0


def avaliar(ender,lista_png,gabarito,nome,indi):
    resultados=[]
    g=[]
    for imagem in range(len(gabarito)):
        if gabarito[imagem][0]+".png" in lista_png:
           
            img =cv2.imread(ender+"\\%s"%gabarito[imagem][0]+".png")
            resultados.append(pdi(img))
 
            g.append(int(gabarito[imagem][indi+1]))
        

    with open(nome,"w") as saida:
        
        saida.write(classification_report(g,resultados))
        saida.write(str((confusion_matrix(g,resultados))))

    return


    return lista_res


banco='E:\\hd1\\rsna-hemorrhage\\genesis-brain-hemorrhage-main\\IDP_resuts\\results'
listas="C:\\Users\\x\\Desktop\\hemo\\lista"
lista_tipos_h=os.listdir("C:\\Users\\x\\Desktop\\hemo\\lista")
lista_png=caminho(banco)


ind=0
for i in  lista_tipos_h :
    lista_txt=abrir(listas+'\\'+i)
    l=lista_contem(lista_txt, lista_png)    
    avaliar(banco,lista_png,lista_txt,'aval_'+i,ind)
    ind=ind+1