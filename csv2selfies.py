from tools import *
import dlib
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
    #x,y,w,h = border( row.x,row.y,row.w,row.h,height, width,0, 3/4)
    x,y,w,h = row.x,row.y,row.w,row.h
    face = rotated_image[y:y+h, x:x+w]

    #face=fix_width(face,new_width=150)

    return face

# read the file
selection_file=sys.argv[1]
images_dir=sys.argv[2]
dest_dir=sys.argv[3]
    
df=pd.read_csv(selection_file)                                        

# sort to show same person
df = df.sort_values(by=["lothar"],ascending=[True])        

compute_encodings=False#True
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



checks=[]
old_enc=[0]
for line, row in enumerate(df.itertuples(), 1):
    print(row.file,row.lothar)
    
    # open for cv
    face = row2face(images_dir,row)

    gray_picture = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    dlib_face = detector(gray_picture)
    print('dlib found ', len(dlib_face),'faces')
    if ( len(dlib_face)>0 ):
        dlib_face=dlib_face[0]
        
        x1 = dlib_face.left() # left point
        y1 = dlib_face.top() # top point
        x2 = dlib_face.right() # right point
        y2 = dlib_face.bottom() # bottom point

        # Look for the landmarks
        landmarks = predictor(image=gray_picture, box=dlib_face)
        x = landmarks.part(27).x
        y = landmarks.part(27).y
        
        # Draw a circle
        cv2.circle(img=face, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    #eyes = eye_cascade.detectMultiScale(gray_picture)
    #print(len(eyes))
    #for (ex,ey,ew,eh) in eyes: 
    #    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

    # select dest dir.
    dest=os.path.join(dest_dir,row.lothar)
    if not os.path.exists(dest):
        os.makedirs(dest)
                  
    # save image
    selfie_name=(row.date+'.jpg')
    cv2.imwrite(os.path.join(dest,selfie_name), face)

    if (compute_encodings):
        # save encoders
        im_pil = Image.fromarray(face)
        im4face_rec = np.array(im_pil.convert('RGB'))

        # compute encoding of current selfie
        x,y,w,h=row.x,row.y,row.w,row.h
        print(row.x,row.y,row.w,row.h)
        encoding = face_recognition.face_encodings(im4face_rec)#,
                                                   #known_face_locations=[[y+h,x+w,y,x]])
        #if (len(old_enc)>2):
        #    print(np.linalg.norm(old_enc-encoding))
        #print(encoding[0][0])
        #old_enc=encoding
        
        if (len(encoding)>0):
            encoder_name=(row.date+'.txt')
            np.savetxt(os.path.join(dest,encoder_name), encoding)
        
