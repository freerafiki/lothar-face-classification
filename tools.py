import face_recognition
import sys
import glob
import numpy as np
import re
from datetime import datetime
import os
import cv2
import PIL
from PIL import Image
import pandas as pd

lothars = ['ago', 'diciommo', 'facca', 'huba', 'lollo',
               'moz', 'paggi', 'palma', 'pecci', 'scotti', 'tonin']

ratio=3/4

additional_width=1/10

def init_lothar_encoders(lothar,path_for_directory='./'):
    """
    Define a list of encoders from multiple images
    stored in (name are those in lothars list).
    Files in each subdirectory can have any name 

    lothar
       img1
       img2
       ...
    """
    encoders=[]
    encs= glob.glob(path_for_directory+lothar+'/*txt')
    for enc_name in encs:
        enc=np.loadtxt(enc_name).flatten()
        encoders.append(enc)
    return encoders

def missing_lothar(Ymd,df):
    """
    Return the list of missing lothar for a given date
    """
    df_monday=df.loc[df['date'] == Ymd ]

    df_monday =  df_monday.columns[df_monday.isnull().any()].tolist()
    
    return list_of_missing_lothar



def rotation_angles(step=5, limit=45):
    """
    Create a list rotation angles
    """
    angles=[0,-90,90]
    for angle in range(step,limit,step):
            for sign in [-1,1]:
                for base in [0,-90,90]:
                    angles.append(base+sign*angle)
    return angles

def compare_image_with_encoders(encoders,
                                image_to_test_encoding,
                                threeshold=0.6):
    """
    Compare one image against multiple encoders

    Args:
    encoders: list of file and corresting encoders
    image_to_test_encoding: image path with just one face
    threeshold : threeshold distance,(default=0.6)

    Returns:
    True or False
    """
    # Load a test image and get encondings for it
    #image_to_test = face_recognition.load_image_file(image_path)
    #image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]


    
    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance(
        encoders, image_to_test_encoding
    )
    avg_distance = sum(face_distances)/len(face_distances)
    if (avg_distance < threeshold):
        return True
    else:
        return False
    
def init_lothars_encoders(lothar_dir):
    """
    Create a list where each element has
    [ lothar_name, [list file_encoders]]
    """
    lothars_enconders=[]
    for lothar in lothars:
        # create list where each element contains ['lothar', encoder]
        lothars_enconders.append([lothar,
                                  init_lothar_encoders(
                                      lothar,
                                      path_for_directory=lothar_dir)])
    return lothars_enconders
            
            
            
        # get date
        
    

def which_lothar(image_to_test_encoding,lothars_encoders,threeshold):
    """
    Return the name of the person in the (encoded) image
    or notfound string
    """
    found=False
    stringout='notfound'
    for lothar_encoders in lothars_encoders:
        result=compare_image_with_encoders(lothar_encoders[1],
                                           image_to_test_encoding,
                                           threeshold=threeshold)
        if result:
            stringout=lothar_encoders[0]
            found=True
            break
    return stringout,found

def str2date(string):
    """"
    Function to define datetime obj from string.
    
    Args: 
        string(str): string with a date. Only
        few format are supported
    """
    try:
        # Telegram
        result = re.search('_(.*)_', string)
        str_date=result[0][1:11]
        date_obj = datetime.strptime(str_date, '%d-%m-%Y')
    except:
        pass

    try:
        # in whatsapp
        result = re.search('-(.*)-', string)
        str_date=result[0][1:9]
        date_obj = datetime.strptime(str_date, '%Y%m%d')
    except:
        pass
    
    return date_obj

def fix_width(im,new_width=600):
    """
    Scale image to get desired width 
    """
    
    # rescale
    old_size = im.shape[:2] # old_size is in (height, width) format
    cur_ratio= old_size[1]/old_size[0] # w/h
    new_height = int(new_width/cur_ratio)  
    im = cv2.resize(im, (new_height, new_width))
    
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_width, new_height))

    return im

def fix_ratio(im,new_ratio=3/4):
    """
    Add black border to get desired ratio width/height
    """
    
    # rescale
    old_size = im.shape[:2] # old_size is in (height, width) format
    cur_ratio= old_size[1]/old_size[0] # w/h
    
    if ( cur_ratio > new_ratio ):
        # h too small
        # delta_w = desired_size - new_size[1]
        new_height= old_size[0]*new_ratio
        delta_h = new_height - old_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = 0, 0
        
    elif (cur_ratio <  new_ratio ):
        # w too small
        new_width= old_size[1]/new_ratio
        delta_w = new_width - old_size[1]
        #delta_h = desired_size - new_size[0]
        top, bottom = 0, 0
        left, right = delta_w//2, delta_w-(delta_w//2)
            
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    
    return new_im



def files2monday(files):
    """
    Find the files with a monday date
    
    Args:
        files (list): list with image path  

    Returns:
        mondays (list) : lsit of image taken at monday
    """

    mondays=[]
    for f in files:
        fname=os.path.basename(f)
        date = str2date(fname)
        if ( date.weekday() == 0):
            mondays.append(f)
            
    return mondays

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result
    
def lothars_in_cv2image(image, lothars_encoders,fc):
    """
    Given image open with opencv finds
    lothars in the photo and the corresponding name and encoding
    """

    # init an empty list for selfie and corresponding name
    lothar_selfies=[]
    names=[]
    encodings=[]
    
    # rgb image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for HaarCascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    
    # cycle angles to until cv find a faces
    found=False
    angles=rotation_angles(5)
    for angle in angles:      
        r_gray=rotate_image(gray,angle)

        faces = fc.detectMultiScale(r_gray,
                                    scaleFactor=1.3,
                                    minNeighbors=6,
                                    minSize=(30, 40),
                                    flags=cv2.CASCADE_SCALE_IMAGE)

        # cycle all faces found
        for i,face  in enumerate(faces):
            # define the face rectangle
            (x,y,w,h) = face
            height, width = image.shape[:2]
            extra_h=((1+2*extra)/ratio-1)/2
            x=int(max(0,x-w*extra))
            y=int(max(0,y-h*extra_h))
            w=int(min(w+2*w*extra,width))
            h=int(min(h+2*h*extra_h,height))

            print('w/h=',w/h)

            # rotate colored image
            rotated_image=rotate_image(image,angle)
            
            # Save just the rectangle faces in SubRecFaces (no idea of meaning of 255)
            #cv2.rectangle(rotated_image, (x,y), (x+w,y+h), (255,255,255))
            sub_face = rotated_image[y:y+h, x:x+w]

            index, name, encoding = lothars_in_selfies([subface], lothars_encoders, num_jitters=2,keep_searching=False)

            if (len(name)>0):
                lothar_selfies.append(sub_face)
                names.append(which_lothar_is)
                encodings.append(encoding)
                found=True

        # break angle changes if a lothar was found
        if (found):
            break
            
    return lothar_selfies, names, encodings

def filename4crop(filename):
    pre, ext = os.path.splitext(filename)
    selfie_name=(pre+'_crop'+str(i)+'.jpg')
    return selfie_name 

def filename4encoding(filename):
    pre, ext = os.path.splitext(filename)
    selfie_name=(pre+'_crop'+str(i)+'.txt')
    return selfie_name

def border(x,y,w,h,height, width, additional_width,ratio):
    extra_w=additional_width
    extra_h=((1+2*extra_w)/ratio-1)/2
    x=int(max(0,x-w*extra_w))
    y=int(max(0,y-h*extra_h))
    w=int(min(w+2*w*extra_w,width))
    h=int(min(h+2*h*extra_h,height))

    return x,y,w,h

def faces_in_cv2image(image,fc,
                      scaleFactor=1.1,
                      keep_searching=False):
    """
    Find faces in photo try angles
    """

    # init an empty list for faces
    faces_found=[]
    face_positions=[]
    
    # rgb image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for HaarCascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # cycle angles to until cv find a faces
    found=False
    angles=rotation_angles(step=4,limit=45) 
    for angle in angles:
        for factor in [1.1]:#,1.2,1.3,1.4]:
            r_gray=rotate_image(gray,angle)
            faces = fc.detectMultiScale(r_gray,
                                        scaleFactor=factor,#scaleFactor,
                                        minNeighbors=6,
                                        minSize=(20, 20),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
            #print('angles',angle,'factor', factor,len(faces))
            if ( len(faces)>0 ):
                found=True
                # rotate colored image
                rotated_image=rotate_image(image,angle)
            
                # cycle all faces found
                for i, face  in enumerate(faces):
                    # define the face rectangle
                    (x,y,w,h) = face
                    face_positions.append([angle,x,y,w,h])

                    # define the face rectangle
                    height, width = image.shape[:2]
                    x,y,w,h = border( x,y,w,h,height, width, additional_width, ratio)
                    
                    
                   
                    
                    # Save just the rectangle faces in SubRecFaces (no idea of meaning of 255)
                    #cv2.rectangle(rotated_image, (x,y), (x+w,y+h), (255,255,255))
                    sub_face = rotated_image[y:y+h, x:x+w]
                    sub_face = fix_width(sub_face,new_width=600)
                    
                    # append face
                    faces_found.append(sub_face)

            # break angle changes if faces where found or
            if (found) and not (keep_searching):
                break
        # break angle changes if faces where found or
        if (found) and not (keep_searching):
            break
            
    return faces_found,face_positions


def lothars_in_cv2selfies(selfies, lothars_encoders,known_face_locations=None, num_jitters=4,keep_searching=False):
    """Given a list of cv2 images (we assume there is only one face per selfie)
    return the lothars in each selfie or 'no' string
    """

    indeces=[]
    names=[]
    encodings=[]
    for i, sub_face, in enumerate(selfies) :
        # convert to the format read by face-recognition
        # Some passages can but saved

        
        img = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(sub_face)
        im4face_rec = np.array(im_pil.convert('RGB'))
    
        # compute encoding of current selfie
        image_to_test_encodings = face_recognition.face_encodings(im4face_rec,
                                                                  known_face_locations=known_face_locations,
                                                                  num_jitters=num_jitters,model='large')
        print('faces_encoding=',len(image_to_test_encodings))
        if (len(image_to_test_encodings) == 1 ):
            # test 
            [which_lothar_is, found_lothar] = which_lothar(image_to_test_encodings[0],lothars_encoders,threeshold=0.6)
            if (found_lothar):
                # append name and encodings
                indeces.append(i)
                names.append(which_lothar_is)
                encodings.append(image_to_test_encodings[0])
            if (not keep_searching):
                break

    return indeces, names, encodings

"""
# get dir with images and create list of all images

directory_known_images=sys.argv[1]
destionation_dir =sys.argv[2]
files = sys.argv[3:]

# select only file with a monday date
mondays=files2monday(files)


#to find path of xml file containing haarCascade file

cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
cfp2 = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
fc = cv2.CascadeClassifier(cfp)
fc2 = cv2.CascadeClassifier(cfp2)

copy=True
search_lothar=True
crop=False
if (search_lothar):
    # create encoders
    lothars_encoders=init_lothars_encoders(directory_known_images)

# cycle all images and append labels to csv file
hs = open("crop_original.csv","w")
#fsel = open("selection.csv","w")
selection_data=[]
nfaces_in_mondays=[]
for file in mondays:

    
    # just the filename
    filename=os.path.basename(file)
    directory=os.path.dirname(file)
    
    # open for cv
    image = cv2.imread(file)

    # detect faces 
    faces, face_positions = faces_in_cv2image(image)

    nfaces_in_mondays.append(len(faces))
    print(len(faces),file)
    if (search_lothar):
        # find lothars
        dest_selfies_in_image=[]
        for i, face in enumerate(faces):
            [index, name, encoding] = lothars_in_cv2selfies([face],
                                                            lothars_encoders,
                                                            keep_searching=False)
            if (True) and (len(name)>0):
                print('Found ' + name[0] +'!!!')
                name=name[0]
                encoding=encoding[0]
                position=face_positions[i]
                alpha,x,y,w,h=position

                date=str2date(filename)
                if (copy):
                    filepath=filename
                    command='cp '+file +' '+os.path.join(destionation_dir)
                    os.system(command)
                else:
                    filepath=file
                                
                selection_data.append([filepath,name,alpha,x,y,w,h,date])

                if (crop):
                    # select dest dir.
                    dest=os.path.join(destionation_dir,name)
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                  
                    # save image
                    pre, ext = os.path.splitext(filename)
                    selfie_name=(pre+'_crop'+str(i)+'.jpg')
                    cv2.imwrite(os.path.join(dest,selfie_name), face)

                    # store path of selfies originated from the image
                    dest_selfies_in_image.append(os.path.join(dest,selfie_name))
                  
                    # save encoders
                    encoder_name=(pre+'_crop'+str(i)+'.txt')
                    np.savetxt(os.path.join(dest,encoder_name),encoding)
                    
                    # add file and sons
                    hs.write(selfie_name+','+file) 

hs.close()


cols = ['file', 'lothar', 'alpha','x','y','w','h','date']
df = pd.DataFrame(selection_data, columns=cols)
# sort first by date and then by lothar name
df = df.sort_values(by=["date","lothar"],ascending=[True,True])       

df.to_csv(os.path.join(destionation_dir,'selection.csv'))

nfiles=len(files)
nmondays=len(mondays)
print('N files=',  nfiles)
print('N mondays=',nmondays)
print('% selfies=',nfaces_in_mondays.count(1)/nmondays*100)
print('% not found=',nfaces_in_mondays.count(0)/nmondays*100)
print('% multiple=',(len(nfaces_in_mondays)-nfaces_in_mondays.count(0)-nfaces_in_mondays.count(1))/nmondays*100)

"""
