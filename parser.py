from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import urllib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
from tqdm import tqdm



dom = ['273', '392', '278', '1342', '50', '624', '1340', '1343', '272', '916', '4138', '534', '164', '3510', '1347', '1348','223', '1346', '3933', '2574', '1911', '3900', '113', '3491', '190', '3541']
dvor = ['776', '2632', '1335', '39', '745', '806', '6011', '101', '120', '2003', '270']
list1=['273', '392', '278', '1342', '50', '624' , '1340', '1343', '272', '916', '4138', '534', '164', '1347', '3933', '2574', '776', '2632', '1335', '745', '806']
list_links=[]
for i in tqdm(list1):
   
    page = "https://www.angrycitizen.ru/?type=ajax&controller=cases&action=filter&problem_id=%s&filter=solved&order=rating&page=1"%i
    r=requests.get(page, timeout=15)
    soup=BeautifulSoup(r.content,'html.parser')
    g=soup.find('div',{'class':'org-paginator'}).contents[-2].text
    for ii in tqdm(range(round(int(g)*0.33))):
        pagei='https://www.angrycitizen.ru/?type=ajax&controller=cases&action=filter&problem_id=%s&filter=solved&order=rating&page=%d'%(i,ii)
        
        r=requests.get(pagei, timeout=15)
        soup=BeautifulSoup(r.content,'html.parser')
        for gg in soup.find_all('div',{'class':'float-left-fixed-pr'}):
            if int(gg.contents[-2].text[-2:])>=17:
                list_links.append(gg.nextSibling.nextSibling.contents[3]['href'])
list_links_2=list(set(list_links))
len(list_links_2)
list_df=[]
alltry=0
falsetry=0
for i in tqdm(list_links_2):
    try:
        page='https://www.angrycitizen.ru%s'%i
        r=requests.get(page, timeout=300)
        soup=BeautifulSoup(r.content,'html.parser')
        problema=soup.find('p',{'class':'pathCaseMain'}).contents[-2].text
        text=soup.find('div',{'class':'case_description'}).contents[0].strip()
        adress=soup.find('h3').get_text()
        list_df.append([text,problema,adress])
    except:
        continue
headers=['text','vid','adress']
df=pd.DataFrame(list_df,columns=headers)

df.to_csv('otzivi_resh.csv',sep=';')
