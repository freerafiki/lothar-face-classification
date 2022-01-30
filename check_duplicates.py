from tools import *
import pandas as pd
import copy

def select_index(duplicates,indeces):
    # form previous images
    im_h = hconcat_resize_min(duplicates)
    cv2.imwrite('out.jpg', im_h)
    which_photo=0
    print('collected=',indeces)
    while True:
        which_photo=int(input('Select one image\n'))
        if ( which_photo<=len(duplicates) and which_photo>=0):
            print('choosen=', indeces[which_photo])
            return indeces[which_photo]


def row2face(row):
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

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    """
    Function to attach images horizonally
    """
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


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
df = copy.copy(df.sort_values(by=["lothar","date"],ascending=[True,True]))

# select duplicated 
#ds=pd.concat(g for _, g in df.groupby(['lothar', 'date']) if len(g) > 1)
dunique=copy.copy(df[~df.duplicated(subset=['lothar', 'date'],keep=False)])
ds=copy.copy(df[df.duplicated(subset=['lothar', 'date'],keep=False)])


#print(dunique[dunique.duplicated(subset=['lothar', 'date'],keep=False)])
unique_indeces=list(dunique.index)#original_indeces.difference(duplicate_indeces)


#print(dunique[dunique.duplicated(subset=['lothar', 'date'],keep=False)])
"""
test=df.iloc[df.index.isin(unique_indeces),:]
print(dunique)
test=test[test.duplicated(subset=['lothar', 'date'],keep=False)]
print(test)
"""

original_indeces=dunique.index
duplicate_indeces=ds.index


# shrink duplicated to one person
if (not all_lothars):
    ds=ds.loc[ds['lothar'] == one_lothar]

#print(ds.duplicated(subset=['lothar', 'date'],keep=False))


# break if no duplciates was found
if (len(ds) == 0 ):
    "No duplicates found"
    exit()

# set for info in photo
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,700)
fontScale              = 3
fontColor              = (255,255,0)
thickness              = 3
lineType               = 2


duplicates=[]
rapresentatives=[]
i=0
indeces=[]
j=0
print('begin selction')
for line, row in enumerate(ds.itertuples()):
    print('index=',row.Index)
    current_date_lothar=str(row.date)+str(row.lothar)
    if (j == 0):
        previous_date_lothar = current_date_lothar
        j=+1
    
    if ( not current_date_lothar == previous_date_lothar ):
        # update
        previous_date_lothar = current_date_lothar
              
        # append index
        index=select_index(duplicates,indeces)
        rapresentatives.append(index)        
        
        # reset
        duplicates=[]
        indeces=[]
        i=0

    face=row2face(row)
    cv2.imwrite('out.jpg', face)
    cv2.putText(face,str(i), 
                (100,700),
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)

    cv2.putText(face,str(row.file), 
                (10,100),
                font, 
                fontScale/3,
                fontColor,
                thickness,
                lineType)
        
    duplicates.append(face)
    indeces.append(row.Index)
    i+=1
        

# form images
index=select_index(duplicates,indeces)
rapresentatives.append(index)

# create the list of good data
if (all_lothars):
    goods = list(unique_indeces)+rapresentatives
    person_duplicates = ds.index
    other_duplicates  = duplicate_indeces.difference(duplicate_indeces)
else:
    print('only:',one_lothar)
    person_duplicates=ds.index
    other_duplicates=duplicate_indeces.difference(person_duplicates)
    goods=list(unique_indeces)+list(other_duplicates)+rapresentatives



dgood = df.iloc[df.index.isin(goods),:]

#drap = df.iloc[df.index[rapresentatives]]
#drap.to_csv('rap.csv',index=False)


# sort to show same person
#dgood = dgood.sort_values(by=["lothar"],ascending=[True])
print('should have no dupli')
print(dgood[dgood.duplicated(subset=['lothar', 'date'],keep=False)])

dgood.to_csv(checked_selection,index=False)


