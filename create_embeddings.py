import os
import face_recognition
import numpy as np
import pdb
import matplotlib.pyplot as plt

IMAGES_TO_CREATE_EMBEDDING = 10
#photo_folder = os.path.join(os.getcwd(), "lothar-face-bot", "lothar-faces")
lothars = ['ago', 'diciommo', 'facca', 'huba', 'lollo', 'moz', 'paggi',
       'palma', 'pecci', 'scotti', 'tonin']
embedding_folder = os.path.join(os.getcwd(), "lothar-face-bot", "lothar-embeddings")

for lothar in lothars:
    #cur_folder = os.path.join(embedding_folder, lothar)
    #images = os.listdir(cur_folder)
    #for image_path in images[:IMAGES_TO_CREATE_EMBEDDING]:
    image_path = os.path.join(embedding_folder, f"{lothar}.jpg")
    image = face_recognition.load_image_file(image_path)
    recognized_lothar = face_recognition.face_encodings(image)
    if recognized_lothar:
        lothar_embedding = recognized_lothar[0]
        np.savetxt(os.path.join(embedding_folder, f"{lothar}.txt"), lothar_embedding)
        emb = np.reshape(lothar_embedding, (16,8))
        plt.imsave(os.path.join(embedding_folder, f"{lothar}_emb.jpg"), emb, cmap="jet")
        print("saved embeddings for", lothar)
    else:
        print("not found, skipped", lothar)
