from tqdm import tqdm
from transliterate import translit, get_available_language_codes
import stringdist
import pandas as pd

dfr=pd.read_csv('resh.csv', delimiter=",",names = ['num','text','vid', 'adress'])
dfr['resh']=1
dfnr=pd.read_csv('neresh.csv', delimiter=",", names = ['num','text','vid', 'adress'])
dfnr['resh']=0
df=pd.concat([dfr,dfnr],names=['num','text','vid', 'adress','resh'])

list_uniqe=df['adress'].tolist()


df_1=pd.read_csv('export-reestrmkd-77-20191001.csv', delimiter=";")
df_2=pd.read_csv('export-reestruo-77-20191001.csv', delimiter=";")



pd.set_option('display.max_columns', 500)

list_uniqe_fias=df_1['address'].tolist()
list_organisation=df_1['management_organization_id'].tolist()
df['UK'] = 'default value'
df['UK_id'] = 'default value'

from tqdm import tqdm
UK_list=[]
for i in tqdm(list_uniqe):
    dist_list=[]
    for g in (list_uniqe_fias):
        dist_list.append(stringdist.levenshtein(str(i).lower(),str(g).lower()))
    indlist=dist_list.index(min(dist_list))
    UK_list.append(list_organisation[indlist])

df=df.dropna()

uk_names=[]
fias_id_list=df_2['id'].tolist()
name_short_list=df_2['name_short'].tolist()
uk_id=df['uk_id'].tolist()

uk_names=[]
for i in range(len(UK_list)):
    for j in range(len(fias_id_list)):
        if (UK_list[i])==(fias_id_list[j]):
            uk_names.append(name_short_list[j])
df['names']=uk_names
