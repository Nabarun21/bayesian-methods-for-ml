
# # Cifar-10 VAE

import sys
sys.path.append("..")
import helpers
helpers.mask_busy_gpus(wait=False)


import tensorflow as tf
import keras
import numpy as np

from keras.layers import Input, Dense, Lambda, InputLayer, concatenate,LeakyReLU,Conv2D,BatchNormalization,MaxPooling2D,Dropout,Flatten,Reshape,Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.losses import mse,binary_crossentropy
from keras.utils import np_utils




from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



print(x_train[0])

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)

#preprocess
x_train=x_train/255.
x_test=x_test/255.
y_train =keras.utils.to_categorical(y_train,10) ### YOUR CODE HERE
y_test = keras.utils.to_categorical(y_test,10)

print(x_train[0])
# First try a regular VAE. Do not condition on classes.




input_shape=x_train.shape[1:]
latent_dim=8
batch_size=64


#posterior sampling using reparametrization trick
def sampling(args):
    t_mean, t_log_var = args
    batch = K.shape(t_mean)[0]
    print(batch)
    dim = K.int_shape(t_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return t_mean + K.exp(0.5 * t_log_var) * epsilon





#Encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x=Conv2D(16,(3,3),padding='same',kernel_initializer='he_uniform')(inputs)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)

       
x=Conv2D(32,(3,3),strides=2,padding='same',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)
    
#x=MaxPooling2D((2,2))(x)
#x=Dropout(0.25)(x)
    
x=Conv2D(64,(3,3),padding='same',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
    
x=Conv2D(64,(3,3),strides=2,padding='same',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)

x=Conv2D(32,(3,3),strides=2,padding='same',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)

#x=MaxPooling2D((2,2))(x)
#x=Dropout(0.25)(x)
x=Flatten()(x)
    
x=Dense(128,name='denseA',kernel_initializer='he_uniform')(x)
x=LeakyReLU(alpha=0.1)(x)
#x=Dropout(0.2)(x)

t_mean=Dense(latent_dim,name='latentA')(x)
t_log_var=Dense(latent_dim,name='latentB')(x)

t=Lambda(sampling, output_shape=(latent_dim,), name='latent')([t_mean, t_log_var])
encoder = Model(inputs, [t_mean, t_log_var, t], name='encoder')
encoder.summary()





#Decoder
latent_inputs = Input(shape=(latent_dim,), name='t_sampling')
x = Dense(128,kernel_initializer='he_uniform')(latent_inputs)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x = Dense(512,kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)

x=Reshape(target_shape=(4,4,32))(x)
x=Conv2DTranspose(64,3,strides=2,padding='same')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Conv2DTranspose(64,3,strides=2,padding='same')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Dropout(0.2)(x)
x=Conv2DTranspose(32,3,padding='same')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
x=Conv2DTranspose(16,3,strides=2,padding='same')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.1)(x)
outputs=Conv2DTranspose(3,3,padding='same',activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()



#Vae
mse=False
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='deepconv_vae_cifar')
reconstruction_loss = mse(inputs, outputs) if mse else K.binary_crossentropy(inputs,outputs)
reconstruction_loss = K.sum(reconstruction_loss,axis=(1,2,3))
kl_div=-0.5*K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var),axis=-1) 
vae_loss = K.mean(reconstruction_loss + kl_div)
vae.add_loss(vae_loss)
vae.compile(optimizer='adadelta',loss=None)




#train
for epoch in range(31):
    vae.fit(x_train,None,epochs=1,batch_size=batch_size,validation_data=(x_test, None))
    
    if epoch%10==0:
        vae.save_weights('cifar_vae_weights'+str(epoch)+'.h5')
        decoder.save_weights('cifar_decoder_weights'+str(epoch)+'.h5')
        encoder.save_weights('cifar_encoder_weights'+str(epoch)+'.h5')

