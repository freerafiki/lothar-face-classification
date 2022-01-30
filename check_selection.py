from tools import *
import pandas as pd

# read the file
selection_file=sys.argv[1]
images_dir=sys.argv[2]
try:
    checked_selection=sys.argv[3]
except:
    checked_selection=selection_file

try:
    one_lothar=sys.argv[4]
    if not one_lothar in lothars:
        print('Person :'+one_lothar+'is not in lothar list')
        exit()
    all_lothars=False
except:
    all_lothars=True
    
df=pd.read_csv(selection_file)                                        

# sort to show same person
df = df.sort_values(by=["lothar"],ascending=[True])        

# Read throught the file
current_lothar='no'

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,700)
fontScale              = 3
fontColor              = (255,255,255)
thickness              = 2
lineType               = 2


checks=[]
for line, row in enumerate(df.itertuples(), 1):
    if (not all_lothars) and (not row.lothar == one_lothar):
        # skip this 
        continue
    lothar=row.lothar
    
    # open for cv
    image = cv2.imread(os.path.join(images_dir,row.file))

    # rotate colored image
    rotated_image=rotate_image(image,row.alpha)
                    
    # Save just the rectangle faces in SubRecFaces (no idea of meaning of 255)
    height, width = image.shape[:2]
    x,y,w,h = border( row.x,row.y,row.w,row.h,height, width, additional_width, 3/4)
    face = rotated_image[y:y+h, x:x+w]

    face=fix_width(face,new_width=600)


    cv2.putText(face,lothar, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
    
    cv2.imwrite('out.jpg', face)

    # read
    label=input('Press Enter if label is right. Otherwise write lothar name or type no\n')
    if (label == ''):
        pass
    else:
        if (label in lothars):
            df.at[row.Index,'lothar']=label
        else:
            df.at[row.Index,'lothar']='no'
    df.at[row.Index,'checked']+=1

df = df.drop(df[~df.lothar.isin(lothars)].index)
    
df.to_csv(checked_selection,index=False)
