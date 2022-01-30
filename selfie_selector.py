from tools import *
import requests


#url = 'https://'
#r = requests.get(url, allow_redirects=True)
#open('monday_selfies.csv', 'wb').write(r.content)

#df_selfies=pd.read_csv('monday_selfies.csv')


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

# set controls of selector
copy=True
search_lothar=True
crop=True


if (search_lothar):
    # create encoders
    lothars_encoders=init_lothars_encoders(directory_known_images)

# cycle all images and append labels to csv file
hs = open("crop_original.csv","w")
#fsel = open("selection.csv","w")
selection_data=[]
nfaces_in_mondays=[]
for file in mondays:    
    # get filename and directory
    filename=os.path.basename(file)
    directory=os.path.dirname(file)

    # check if we are interested at this date
    """
    Ymd=str2date(filename)
    missing=missing_lothar(Ymd,df_selfeis)
    if (len(missing)==0):
        # Skip the remaing part. All selfies are covered.
        continue
    else:
        # Create a list of encoders that will look for
        encoders_to_use=[]
        for enc in lothars_encoders:
            if (enc[0] in missing):
                encoders_to_use.append(enc)
    """
    encoders_to_use=lothars_encoders
    
    # open for cv
    image = cv2.imread(file)

    # detect faces 
    faces, face_positions = faces_in_cv2image(image,fc)

    nfaces_in_mondays.append(len(faces))
    print(len(faces),file)
    if (search_lothar):
        # find lothars
        for i, face in enumerate(faces):
            [index, name, encoding] = lothars_in_cv2selfies([face],
                                                            encoders_to_use,
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
                pre, ext = os.path.splitext(filename)
                selfie_name=(pre+'_crop'+str(i)+'.jpg')
                                
                selection_data.append([filepath,selfie_name,alpha,x,y,w,h,name,date,0])

                if (crop):
                    # select dest dir.
                    dest=os.path.join(destionation_dir,name)
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                  
                    # save image
                    pre, ext = os.path.splitext(filename)
                    selfie_name=(pre+'_crop'+str(i)+'.jpg')
                    cv2.imwrite(os.path.join(dest,selfie_name), face)
                  
                    # save encoders
                    encoder_name=(pre+'_crop'+str(i)+'.txt')
                    np.savetxt(os.path.join(dest,encoder_name),encoding)
                    
                    # add file and sons
                    hs.write(selfie_name+','+file) 

hs.close()


cols = ['file','crop_file','alpha','x','y','w','h','lothar','date','checked']
df = pd.DataFrame(selection_data, columns=cols)
# sort first by date and then by lothar name
df = df.sort_values(by=["date","lothar"],ascending=[True,True])       

df.to_csv(os.path.join(destionation_dir,'selection.csv'),index=False)

nfiles=len(files)
nmondays=len(mondays)
print('N files=',  nfiles)
print('N mondays=',nmondays)
print('% selfies=',nfaces_in_mondays.count(1)/nmondays*100)
print('% not found=',nfaces_in_mondays.count(0)/nmondays*100)
print('% multiple=',(len(nfaces_in_mondays)-nfaces_in_mondays.count(0)-nfaces_in_mondays.count(1))/nmondays*100)

