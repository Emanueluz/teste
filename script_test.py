
import csv
import torch
import torchvision.models as models
import os 
import cv2

from datasets.rsnaDatasetImage import generate_transforms, generate_dataset, generate_loader


from models import initialize_pretrained_model

#abrir o arquivo 
def abrir(arq_csv):
    with open(arq_csv, 'r') as ficheiro:
        reader = csv.reader(ficheiro)
        reader= list(reader)
        
    return reader



def caminho(arq):
    lista_de_arq=os.listdir(arq)
    return lista_de_arq


#caminho dos asquivos 
arq_csv=r"E:\hd1\rsna-hemorrhage\genesis-brain-hemorrhage-main\data\rsna_\test.csv"
arq_csv=abrir(arq_csv)
dd_csv=arq_csv[0]
print(dd_csv)
del(arq_csv[0])


modelo=r"E:\hd1\rsna-hemorrhage\genesis-brain-hemorrhage-main\models\trainned_models"

modelos=caminho(modelo)



#for n_ima in arq_csv:
imagem=r"E:\hd1\rsna-hemorrhage\genesis-brain-hemorrhage-main\data\rsna\png\triple_window%s"%arq_csv[0][0]
img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)
caminho_model=(r"E:\hd1\rsna-hemorrhage\genesis-brain-hemorrhage-main\models\trainned_models\%s"%modelos[0])
model=torch.load(caminho_model)
device = torch.device('cpu')
model=model.to(device)



nome_modelo=modelos[0][:-3]
num_class=0
model_name=0

_,tamanho_img=initialize_pretrained_model(nome_modelo,num_class)
a=[]

(val_transform)=generate_transforms(tamanho_img,is_train=False)


treinamento_datas=generate_dataset(caminho_model, arq_csv, file_names, transforms)
'''
#carregando o modelo 
for i in modelos:
    model = torch.load(i)
    arq_saida(model,"saida")'''