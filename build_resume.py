from tools import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# read the file
selection_file=sys.argv[1]
resume_file=sys.argv[2]

df=pd.read_csv(selection_file)

# sort to show same person
df = df.sort_values(by=["date","lothar"],ascending=[True,True])

#dates = list(set( df["date"] ))

print(df)

df['date'].astype(str)
df['file'].astype(str)
df['lothar'].astype(str)
table=df.pivot_table(index='date', columns='lothar', values='file', aggfunc='count',fill_value=0)

table_file=df.pivot_table(index='date', columns='lothar', values='file', aggfunc='first', fill_value='')

table_file.to_csv(resume_file)


table.index = pd.DatetimeIndex(table.index)
table_plus=table.resample('7D').sum()
print(table_plus)

fig, ax = plt.subplots(figsize=(9, 11))
#my_colors=[(0.2,0.3,0.3),(0.4,0.5,0.4),(0.1,0.7,0),(0.1,0.7,0)]
my_colors=['red','green','yellow']
sb.heatmap(table_plus,cmap=my_colors,vmin=0,vmax=2,linewidth=0.1, linecolor='k',)
plt.show()


"""
table_crop=df.pivot_table(index='date', columns='lothar', values='subface', aggfunc='first',fill_value='')


list_subface=[str('sub_'+i) for i in lothars]
#print(list_subface)
#list_subface=list_subface.insert(0,str('date'))
print(list_subface)
print(table_crop.columns)
#table_crop.columns=list_subface 
table_crop = table_crop.set_axis(list_subface , axis=1, inplace=False)
print(table_crop)
"""

table.to_csv('resume.csv')



"""
for monday in dates:
    # select row with current monday
    df_monday=df.loc[(df['date'] == monday)]
    files=[]
    crop=[]
    for row in df_monday.itertuples():
        df_monday.loc[(df['lothar'] == lothar)]
        
"""     
