from tools import *
import pandas as pd

def row2face(images_dir,row):
    """
    Convert the information in a row of the dataframe into a image
    """

    # open for cv
    image = cv2.imread(os.path.join(images_dir,row.file))

    # rotate colored image
    rotated_image=rotate_image(image,row.alpha)
                    
    # Save just the rectangle faces in SubRecFaces (no idea of meaning of 255)
    height, width = image.shape[:2]
    x,y,w,h = border( row.x,row.y,row.w,row.h,height, width, additional_width, 3/4)
    face = rotated_image[y:y+h, x:x+w]

    face=fix_width(face,new_width=600)

    return face

# read the file
selection_file=sys.argv[1]
images_dir=sys.argv[2]
dest_dir=sys.argv[3]
    
df=pd.read_csv(selection_file)                                        

# sort to show same person
df = df.sort_values(by=["lothar"],ascending=[True])        


checks=[]
for line, row in enumerate(df.itertuples(), 1):
    print(row.file)
    
    # open for cv
    face = row2face(images_dir,row)

    # select dest dir.
    dest=os.path.join(dest_dir,row.lothar)
    if not os.path.exists(dest):
        os.makedirs(dest)
                  
    # save image
    selfie_name=(row.date+'.jpg')
    cv2.imwrite(os.path.join(dest,selfie_name), face)

    """
    # save encoders
    im_pil = Image.fromarray(face)
    im4face_rec = np.array(im_pil.convert('RGB'))
    
    # compute encoding of current selfie
    encoding = face_recognition.face_encodings(im4face_rec)
    encoder_name=(row.date+'.txt')
    np.savetxt(os.path.join(dest,encoder_name), encoding)
    """
