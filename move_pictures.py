import pandas as pd
import pdb
import os

path_from = os.path.join(os.getcwd(), "photos")
path_to = os.path.join(os.getcwd(), "lothar-face-bot", "lothar-faces")
labels = pd.read_csv('labels.csv')
lothars = ['ago', 'diciommo', 'facca', 'huba', 'lollo', 'moz', 'paggi',
       'palma', 'pecci', 'scotti', 'tonin']
for label, file in zip(labels['lothar'].values, labels['file'].values):
    #pdb.set_trace()
    if label in lothars:
        source_path = os.path.join(path_from, file)
        if os.path.exists(source_path):
            target_folder = os.path.join(path_to, label)
            if os.path.exists(target_folder):
                target_path = os.path.join(target_folder, file)
                print(f"moving {source_path} to {target_path}")
                os.rename(source_path, target_path)
            else:
                print(f"no folder for {label}")
        else:
            print(f"no soure file, probably already moved")
