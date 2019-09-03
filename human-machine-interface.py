#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:11:09 2019

@author: Soriba
"""
import os
import numpy as np
from keras.models import load_model
from PIL import ImageTk, Image, ImageDraw, ImageChops
from process import get_px, get_burst_start

#if you use python2 or python3
try:                        
    import tkinter as tk    
    import tkinter.filedialog as tkfd
except ImportError:
    import Tkinter as tk
    import tkFileDialog as tkfd 


if __name__ == '__main__':
    #We load the architecture and the weights of the pretrained model
    model=load_model('model.h5')      
    """
    Function to draw a rectangle from the coordinates
    """
    def drawrect(drawcontext, xy, outline=None, width=0):   
        (x1, y1), (x2, y2) = xy
        points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
        drawcontext.line(points, fill=outline, width=width)


    def forget():
            CB1.destroy()
            CB2.destroy()
            
            
    """
    Function to detect the white border of the image
    """
    
    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            # in the form (x1, y1, x2, y2)
            return bbox           
        
    """
    Function to load an image from the file system and make a prediction
    """
    
    def select_image():
        global panelA, Label
        #the user specify the path to the spectrogram he wants to predict
        path=tkfd.askopenfilename(initialdir = "/mnt/disque2/IMT/base", 
                                  title = "Select a spectrogram",
                        filetypes = (("png files","*.png"),
                                     ("all files","*.*"))) 
        
        
        filename=os.path.basename(path)
        #file containing the starts of the bursts
        filename_txt=filename.split('.')[0]+'.wav.txt'  
        classes = np.load('classes.npy')
        #class of the image (name of the folder in which it is)
        groundtruth = os.path.basename(os.path.dirname(path)) 
        
        
        if path:
            #we adapt the size of the image for the neural network
            img=Image.open(path).resize((709,532)) 
            #we convert the image into a matrix
            pix=np.array(img)
            #we remove the white border
            box = trim(img)    
            max_x=box[2]-box[0]+1  
            max_y=box[3]-box[1]+1
            print('max_x : {}'.format(max_x))
            bandwidths_px, burst_durations_px = get_px(max_x,max_y)
            #starts of the bursts
            burst_starts = get_burst_start(groundtruth, filename_txt, max_y, 
                                           box[1]) 
            burst_len = burst_durations_px[np.where(classes==groundtruth)[0][0]] 
            #f0 = 0, it corresponds to the center of the image (on the x axis)
            f0_px=box[0] + np.ceil(max_x/2)  
            bw=bandwidths_px[np.where(classes==groundtruth)[0][0]]
            #normalization
            processed = pix/255 
            #prediction of the model
            prediction = model.predict_classes(processed[np.newaxis,:])[0] 
            
        if var1.get()==1:
            for k in range(len(burst_starts)):
                draw = ImageDraw.Draw(img) 
                drawrect(
                        draw, 
                         [(f0_px-bw/2, burst_starts[k]), 
                          (f0_px+bw/2, burst_starts[k]+burst_len)],
                          outline="blue",
                          width=2
                          )
                
        display = ImageTk.PhotoImage(img)
        #initialization of panelA
        if panelA is None: 
            #panelA will display the image
            panelA = tk.Label(image=display) 
            panelA.image = display
            panelA.pack(side='bottom', padx=10, pady=10)
            
        else:
            #if panelA is already initialized
            panelA.configure(image=display) 
            panelA.image=display
            forget()
            
        #prediction on a green background if it is correct, red otherwise   
        bg='green' if classes[prediction] == groundtruth  else 'red'  
        
        text = "Predicted class : {} \nTrue class : {}".format(classes[prediction],
                                  groundtruth) 
        
        if Label is None: 
            Label= tk.Label(top_frame, text=text,
                       fg='white', bg=bg)
            Label.pack(fill='x')
        
        else:
            Label.configure(text=text, bg=bg)
        
    #main window
    root=tk.Tk() 
    root.title('application')
    top_frame = tk.Frame(root).pack()
    bottom_frame = tk.Frame(root).pack(side = "bottom")
    
    
    panelA=None
    Label=None
    var1=tk.IntVar()
    var2=tk.IntVar()
    
    CB1=tk.Checkbutton(bottom_frame, text = "groundtruth detection",
                       variable=var1, command=forget)
    CB1.pack(side = 'right')
    CB2=tk.Checkbutton(bottom_frame, text = "predicted detection",
                       variable=var2, command=forget)
    CB2.pack(side= 'right')
    btn = tk.Button(bottom_frame, text = 'Choose a spectrogram', 
                    command=select_image)
    btn.pack(side='bottom', fill='both', expand='yes', padx='10', pady='10')
    
    root.mainloop()