
import numpy as np 
import pandas as pd 
import os
from skimage.io import imread 
from PIL import Image 
from glob import glob
from sklearn.preprocessing import LabelEncoder
import cv2
from tqdm import tqdm
from keras.models import load_model
from Tkinter import *
import Tkinter as tk
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import os

stra = ''
    
def select_image():
	# grab a reference to the image panels
	global panelA
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkFileDialog.askopenfilename()
	print (path)
        img_data = imread_and_normalize(path)
        n_image = np.expand_dims(img_data,0)
        s=np.argmax(model.predict(n_image)[0])
        print (cat_encoder.classes_[s])
        stra = cat_encoder.classes_[s]
        var.set("The Source device is \""+stra+"\"")
	cmd = 'espeak "{0}" 2>/dev/null'.format(stra)
	os.system(cmd)
	# ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		
 
		# OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
		# convert the images to PIL format...
		image = Image.fromarray(image)

 
		# ...and then to ImageTk format
		image = ImageTk.PhotoImage(image)

		# if the panels are None, initialize them
		if panelA is None :
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="bottom", padx=10, pady=10)
 

 
		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image)
			panelA.image = image
		

def get_class_from_path(filepath):
    return os.path.dirname(filepath).split(os.sep)[-1]

def imread_and_normalize(im_path):
    img_data = pil_imread(im_path)
    img_data = cv2.cvtColor(img_data[:,:,[2,1,0]], cv2.COLOR_BGR2LAB)
    img_data[:,:,0] = clahe.apply(img_data[:,:,0])
    img_data = cv2.cvtColor(img_data, cv2.COLOR_LAB2BGR)
    
    return (img_data.astype(np.float32))/255.0

root = tk.Tk()

#root.configure(background='blue')



r=root.geometry('1000x1000')
root.title('CAMERA DETECTION FROM IMAGE')
panelA = None

frame = tk.Label(root, text="A Simple Application to findout the Camera from image")

#bg image frame
filename = PhotoImage(file = "pi.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
##########
frame.pack()

btn = tk.Button(root, text="Select the image", command=select_image,fg="white",bg="GREEN",width=200,height=1,font=("Helevetica", 15))
btn.pack(side="bottom", fill="x", expand="no", padx="10", pady="10")

var = StringVar()
label = Label( root, textvariable=var, relief=RAISED,bg="ORANGE",fg="white",width=200,height=2,font=("Lithograph 18  bold") )

var.set("No images Selected")
label.pack()






list_train = glob(os.path.join('train', '*', '*.jpg'))

pil_imread = lambda c_file: np.array(Image.open(c_file))

full_train_df = pd.DataFrame([{'path': x, 'category': get_class_from_path(x)} for x in list_train])
cat_encoder = LabelEncoder()
cat_encoder.fit(full_train_df['category'])
nclass = cat_encoder.classes_.shape[0]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))

model = load_model('weights.best.hdf5')

root.mainloop()



