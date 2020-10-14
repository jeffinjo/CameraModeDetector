
import numpy as np  
import pandas as pd 
import os
from skimage.io import imread # read image
from PIL import Image 
from skimage.exposure import equalize_adapthist
from glob import glob
from sklearn.preprocessing import LabelEncoder
import cv2
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate
from sklearn.model_selection import train_test_split
from keras.models import Sequential

def get_class_from_path(filepath):
    return os.path.dirname(filepath).split(os.sep)[-1]

def imread_and_normalize(im_path):
    img_data = pil_imread(im_path)
    img_data = cv2.cvtColor(img_data[:,:,[2,1,0]], cv2.COLOR_BGR2LAB)
    img_data[:,:,0] = clahe.apply(img_data[:,:,0])
    img_data = cv2.cvtColor(img_data, cv2.COLOR_LAB2BGR)
    return (img_data.astype(np.float32))/255.0

def read_chunk(im_path, n_chunk = 5, chunk_x = 96, chunk_y = 96):
    img_data = imread_and_normalize(im_path)
    img_x, img_y, _ = img_data.shape
    out_chunk = []
    for _ in range(n_chunk):
        x_pos = np.random.choice(range(img_x-chunk_x))
        y_pos = np.random.choice(range(img_y-chunk_y))
        out_chunk += [img_data[x_pos:(x_pos+chunk_x), y_pos:(y_pos+chunk_y),:3]]
    return np.stack(out_chunk, 0)

def generate_even_batch(base_df, sample_count = 1, chunk_count = 5):
    while True:
        cur_df = base_df.groupby('category').apply(lambda x: x[['path']].sample(sample_count)).reset_index()
        x_out = np.concatenate(cur_df['path'].map(lambda x: read_chunk(x, n_chunk=chunk_count)),0)                             
        y_raw = [x for x in cur_df['category'].values for _ in range(chunk_count)]
        y_out = to_categorical(cat_encoder.transform(y_raw))
        yield x_out, y_out

def gap_drop(in_layer): 
    gap_layer = GlobalAveragePooling2D()(Convolution2D(16, kernel_size = 1)(in_layer))
    gmp_layer = GlobalMaxPool2D()(Convolution2D(16, kernel_size = 1)(in_layer))
    return Dropout(rate = 0.5)(concatenate([gap_layer, gmp_layer]))

def create_model():
    inp = Input(shape=(None, None, 3))
    norm_inp = BatchNormalization()(inp)
    gap_layers = []
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)    
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    gap_layers += [gap_drop(img_1)]
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    gap_layers += [gap_drop(img_1)]    
    gap_cat = concatenate(gap_layers)    
    dense_1 = Dense(32, activation=activations.relu)(gap_cat)
    dense_1 = Dense(nclass, activation='softmax')(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(lr=1e-3) # karpathy's magic learning rate
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc']) 
    model.summary
    
    return model  
                   
                  

   


pil_imread = lambda c_file: np.array(Image.open(c_file))
list_train = glob(os.path.join('train', '*', '*.jpg'))
print 'Train Files found', len(list_train)

full_train_df = pd.DataFrame([{'path': x, 'category': get_class_from_path(x)} for x in list_train])
cat_encoder = LabelEncoder()
cat_encoder.fit(full_train_df['category'])
nclass = cat_encoder.classes_.shape[0]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))

train_df, test_df = train_test_split(full_train_df,test_size = 0.15,random_state = 2018,stratify = full_train_df['category']) 
                                     
train_gen = generate_even_batch(train_df, 3, chunk_count = 3)
test_gen = generate_even_batch(test_df, 10)                                  
                                    
(test_x, test_y) = next(test_gen)
model = create_model()
file_path="weights.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_acc", mode="max", patience=400)
callbacks_list = [checkpoint, early] #early

history = model.fit_generator(train_gen, 
                              steps_per_epoch = 10,
                              validation_data = (test_x, test_y), 
                              epochs = 500, 
                              verbose = True,
                              workers = 2,
                              use_multiprocessing = False,
                              callbacks = callbacks_list)



import sys
orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f
print("ACCURACY:")
print(history.history['acc']  )
print('\n')
print("LOSS:")
print(history.history['loss']  )
print('\n')
print("VALUE ACCURACY:")
print(history.history['val_acc']  )
print('\n')
print("VALUE LOSS:")
print(history.history['val_loss']  )
sys.stdout = orig_stdout
f.close()
