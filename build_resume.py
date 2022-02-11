from tools import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def split_years(dt):
    dt['year'] = dt.index.year
    ds=[dt[dt['year'] == y] for y in dt['year'].unique()]
    return ds


# read the file
selection_file=sys.argv[1]
resume_file=sys.argv[2]

df=pd.read_csv(selection_file)

# sort to show same person
df = df.sort_values(by=["date","lothar"],ascending=[True,True])

df['date'].astype(str)
df['file'].astype(str)
df['lothar'].astype(str)


table_file=df.pivot(index='date', columns='lothar', values='file')
table_file.to_csv(resume_file)

#table_file=table_file.reset_index()
#print(table_file)

table_file=df.pivot_table(index='date', columns='lothar', values='file', aggfunc='count',fill_value=0)

table_file.to_csv('out.csv')
table_plus=pd.read_csv('out.csv')

table_plus.index = pd.DatetimeIndex(table_plus.date)
table_plus=table_plus.resample('7D').sum()#.reset_index(inplace=True)
#table_plus.index = pd.DatetimeIndex(table_plus.index)

sdf=split_years(table_plus)
for i, ds in enumerate(sdf):
    del ds['year']
    sb.set()
    fig, ax = plt.subplots()
    my_colors=['red','green','yellow']
    hm=sb.heatmap(ds,cmap=my_colors,vmin=0,vmax=2,
                  linewidth=0.1, linecolor='k',ax=ax)
    hm.set_yticklabels(ds.index.strftime('%Y-%m-%d'))
  
    figure = hm.get_figure()    
    figure.savefig('covered'+str(2013+i)+'.png')




