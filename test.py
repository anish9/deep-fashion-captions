import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.layer_blocks import *
import matplotlib.pyplot as plt

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


im_path = sys.argv[1]
image_size = (320,320,3)
seq_max_length = 100
vectorizer = restore_tokenizer("weights/fashion500vect")


caption_nn = load_model("weights/fashion_caption.h5",custom_objects={"positional_embed":positional_embed,
                                                "encoder_model":encoder_model,
                                                "decoder_model":decoder_model})

def image_prep(imagefile,image_size):
    imagefile = tf.io.read_file(imagefile)
    imarray = tf.image.decode_jpeg(imagefile)
    imarray =  tf.image.resize_with_pad(imarray,target_height=image_size[0],target_width=image_size[1])
    imarray = tf.expand_dims(imarray,axis=0)
    return imarray,tf.cast(imarray[0,:,:,:],tf.uint8)

def predict_caption(array,nnet,vectorizer,seq_length):
    output_seq = " "
    start_tag,end_tag = "<sos>","<eos>"
    output_seq+=start_tag
    target_vocab =vectorizer.get_vocabulary()
    target_vocab_inv = dict(zip(range(len(target_vocab)),target_vocab))
    for i in range(seq_length):
        current  = output_seq
        phrase = vectorizer([current])[:,:-1]
        prediction = nnet.predict((array,phrase),verbose=0)
        
        prediction_to_word =target_vocab_inv[np.argmax(prediction[0,i,:])]
        if prediction_to_word==end_tag:
            break
        output_seq+=" "+prediction_to_word
    
    final_sequence = " ".join(output_seq.split()[1:])
        
    return final_sequence+"."

image_input,rgb_image = image_prep(im_path,image_size)

plt.figure(figsize=(10,8))
plt.imshow(rgb_image)
text_out = predict_caption(image_input,nnet=caption_nn,vectorizer=vectorizer,seq_length=100)
print("-"*100)
print(f"Generated Caption : {text_out}")
print("-"*100)
plt.xlabel(text_out)
plt.show()