#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:11:09 2019

@author: Soriba
"""
# In[1]:
#Imports
import os
import numpy as np
from keras.models import load_model
from cnn import process_data, evaluate
from PIL import ImageTk, Image, ImageDraw, ImageChops
from process import *

try:                        # En fonction de Python 2 ou 3
    import tkinter as tk    
    import tkinter.filedialog as tkfd
except ImportError:
    import Tkinter as tk
    import tkFileDialog as tkfd 



# In[ ]:
model=load_model('model.h5')  #On charge l'architecture et les poids du modèle qui a été préalablement entrainé
#X_5000, y_5000 = process_data(data_path='5000msPNG')
#evaluate(model, X_5000, y_5000)   
# In[ ]:
    
if __name__ == '__main__':
    
    """
    Fonction permettant de dessiner un rectangle à partir des coordonnées
    """
    def drawrect(drawcontext, xy, outline=None, width=0):   #permet de tracer des rectangles sur l'image
        (x1, y1), (x2, y2) = xy
        points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
        drawcontext.line(points, fill=outline, width=width)


    def forget():
            CB1.destroy()
            CB2.destroy()
            
            
    """
    Fonction permettant de detecter la bordure blanche de l'image
    """
    
    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return bbox           # de la forme (x1, y1, x2, y2)
        
    """
    Fonction permettant de charger une image et de faire une prédiction
    """
    
    def select_image():
        global panelA, Label
        path=tkfd.askopenfilename(initialdir = "/mnt/disque2/IMT/base", title = "Selectionner un spectrogramme",
                        filetypes = (("Fichiers png","*.png"),("Tous les fichiers","*.*"))) #L'utilisateur doit spécifier le chemin vers le spectrogramme qu'il souhaite prédire
        
        
        filename=os.path.basename(path)
        filename_txt=filename.split('.')[0]+'.wav.txt'  #fichier contenant les debuts des bursts
        classes = np.load('classes.npy')
        groundtruth = os.path.basename(os.path.dirname(path)) #vraie classe de l'image (nom du dossier dans lequel elle se trouve)
        
        
        if path:
            img=Image.open(path).resize((709,532)) #On adapte la taille de l'image à l'entrée du réseau
            pix=np.array(img) # On convertit l'image en pixels
            box = trim(img)  #vraie taille du spectrogramme (en enlevant les bandes blanches)  
            max_x=box[2]-box[0]+1  #par exemple si ça va des pixels 2 à 4, il y'a 3 pixels
            max_y=box[3]-box[1]+1
            print('max_x : {}'.format(max_x))
            bandwidths_px, burst_durations_px = get_px(max_x,max_y)
            burst_starts = get_burst_start(groundtruth, filename_txt, max_y, box[1]) # debuts des bursts pour l'image en question
            burst_len = burst_durations_px[np.where(classes==groundtruth)[0][0]] #importé de 'process.py'
            f0_px=box[0] + np.ceil(max_x/2)  #f0 = 0, cela correspond au milieu de l'image, il faut trouver la bonne
            bw=bandwidths_px[np.where(classes==groundtruth)[0][0]]
            processed = pix/255 #mise au format numpy et normalisation
            prediction = model.predict_classes(processed[np.newaxis,:])[0] #prédiction de la classe de l'image avec le modèle chargé précédement
            
        if var1.get()==1:
            for k in range(len(burst_starts)):
                draw = ImageDraw.Draw(img) #problème avec le protocole C car il n'a pas la même résolution fréquentielle que les autres
                drawrect(draw, [(f0_px-bw/2, burst_starts[k]), (f0_px+bw/2, burst_starts[k]+burst_len)], outline="blue", width=2)
        
        
        display = ImageTk.PhotoImage(img)
        
        if panelA is None: #à l'initialisation de panelA
            panelA = tk.Label(image=display) #panelA va permettre d'afficher l'image
            panelA.image = display
            panelA.pack(side='bottom', padx=10, pady=10)
            
        else:
            panelA.configure(image=display) #si panelA est déja initialisé
            panelA.image=display
            forget()
            
            
        bg='green' if classes[prediction] == groundtruth  else 'red'  #prediction sur fond vert si elle est correcte, rouge sinon
        
        text = "Classe prédite : {} \nVraie classe : {}".format(classes[prediction], groundtruth) # text à afficher en fonction de la prédiction et de la vraie classe de l'image
        
        if Label is None: #le label contiendra le texte à afficher, condition suivant le même principe que panelA
            Label= tk.Label(top_frame, text=text,
                       fg='white', bg=bg)
            Label.pack(fill='x')
        
        else:
            Label.configure(text=text, bg=bg)
        
            
        
    root=tk.Tk() #fenêtre principale
    root.title('application')
    top_frame = tk.Frame(root).pack()
    bottom_frame = tk.Frame(root).pack(side = "bottom")
    
    
    panelA=None
    Label=None
    var1=tk.IntVar()
    var2=tk.IntVar()
    
    CB1=tk.Checkbutton(bottom_frame, text = "groundtruth detection",variable=var1, command=forget)
    CB1.pack(side = 'right')
    CB2=tk.Checkbutton(bottom_frame, text = "predicted detection", variable=var2, command=forget)
    CB2.pack(side= 'right')
    btn = tk.Button(bottom_frame, text = 'Choisir un spectrogramme', command=select_image)
    btn.pack(side='bottom', fill='both', expand='yes', padx='10', pady='10')
    
    root.mainloop()
    

    
    
    
    
    
    
    
    