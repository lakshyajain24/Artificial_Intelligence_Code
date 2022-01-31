
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.contrib.operators.ssh_operator import SSHOperator
from airflow.operators.bash_operator import BashOperator
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import date
import os
from os.path import exists
import psycopg2
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine


# ================================custom functions=============================

def connect_mongo_postgress():
    client = MongoClient(host= "192.168.0.8", port=27017,username='root',password='rootpassword')
    mongodata = client.answers
    users_mongo = mongodata.data
    df= pd.DataFrame(list(users_mongo.find()))
    
    
    #df = pd.read_csv("dags/answers.csv",encoding = "ISO-8859-1")
    dict1 = {}
    for i in range (len(df)):
        
        if df['question'][i] not in dict1:
            temp = {}
            for j in range(len(df['createdAt'])):
                if df['question'][j]==df["question"][i]:
                    if df['createdAt'][j].split("T")[0] not in temp:
                        temp[df['createdAt'][j].split("T")[0]] = [df["Body."][j]]
                       
                    else:
                        temp[df['createdAt'][j].split("T")[0]].append( df["Body."][j])
                
            dict1[df['question'][i]] = temp
    
    
    df_new = pd.DataFrame(dict1)       
    df_new.reset_index(level=0, inplace=True)
    df_new.rename(columns={'index':'Date'}, inplace=True)
    for col in dict1.keys():
        df_new[col] = df_new[col].map(lambda a: " , ".join(a),na_action='ignore')
    
    #Connection with Postgresql
    conn_string = 'postgresql://airflow:airflow@192.168.0.8/airflow'
    db = create_engine(conn_string)
    conn = db.connect()
    
    df_new.to_sql('data', con=conn, if_exists='replace',
    		index=False)
    conn = psycopg2.connect(conn_string
    						)
    conn.autocommit = True
    conn.commit()
    conn.close()

    #Building Function
    def remove_stopwords(string): 
        word_tokens = word_tokenize(string)
        stop_words = list(set(stopwords.words('english')))+['.','0','1','2','3','4','5','6','7','8','9','missing']
        list1 = [w for w in word_tokens if not w.lower() in stop_words]
        str1 = " "
        list1 = []
        for w in word_tokens:
            if w not in stop_words:
                list1.append(w)
        string = str1.join(list1)
        return(string)
    
    #Building Function
    def word_count(str):
        counts = dict()
        words = str.split()
    
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts
    
    df_new1=df_new.copy()
    df_new1.fillna("missing", inplace=True)
    freqdf = pd.DataFrame(columns=df_new1.columns)
    
    n = 0
    int(n)
    for i in range(0,len(df_new1["Date"])):
        if(n>0):
            freqdf.loc[df_new1.index[n]] = df_new1.iloc[n]+", "+freqdf.iloc[n-1]
        else:
            freqdf.loc[df_new1.index[n]] = df_new1.iloc[n]
        n=n+1
    del freqdf["Date"]
    
    
    for i in freqdf.columns:
        t=0
        for j in freqdf[i]:
            if(isinstance(j,float) or isinstance(j, int)):
                pass        
            else:
                new = word_count(remove_stopwords(j))
                freqdf[i][t]=new
            t=t+1
    
    
        
    for i in freqdf.columns:
        for j in freqdf[i]:
            if('missing' in j):
                j.pop('missing')
            elif(',' in j):
                j.pop(',')
            else:
                pass
    
    
    freqdf.insert(loc = 0,
              column = 'Date',
              value = df_new1["Date"])
    
    for i in freqdf.columns:
        freqdf[i] = freqdf[i].astype(str)            
                
    
    
    conn_string = 'postgresql://airflow:airflow@192.168.0.8/airflow'
    db = create_engine(conn_string)
    conn = db.connect()
    
    freqdf.to_sql('frequency', con=conn, if_exists='replace',
    		index=False)
    conn = psycopg2.connect(conn_string
    						)
    conn.autocommit = True
    conn.commit()
    conn.close()

    return "Mongo Postgress Done"


#===========================================================

def connect_minio():

    client = MongoClient(host= "192.168.0.8", port=27017,username='root',password='rootpassword')
    mongodata = client.answers
    users_mongo = mongodata.data
    df= pd.DataFrame(list(users_mongo.find()))

    #df = pd.read_csv("dags/answers.csv",encoding = "ISO-8859-1")
    dict1 = {}
    for i in range (len(df)):
        
        if df['question'][i] not in dict1:
            temp = {}
            for j in range(len(df['createdAt'])):
                if df['question'][j]==df["question"][i]:
                    if df['createdAt'][j].split("T")[0] not in temp:
                        temp[df['createdAt'][j].split("T")[0]] = [df["Body."][j]]
                       
                    else:
                        temp[df['createdAt'][j].split("T")[0]].append( df["Body."][j])
                
            dict1[df['question'][i]] = temp
    

    def n_grams(text,ngram=1):
        stop_words = list(set(stopwords.words('english')))+['0','1','2','3','4','5','6','7','8','9','missing',',']
        words=[word for word in text.split(" ") if word not in stop_words]
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=[' '.join(ngram) for ngram in temp]
        return ans

    df_new2 = pd.DataFrame(dict1)       
    df_new2.reset_index(level=0, inplace=True)
    df_new2.rename(columns={'index':'Date'}, inplace=True)
    for col in dict1.keys():
        df_new2[col] = df_new2[col].map(lambda a: " , ".join(a),na_action='ignore')

    #Creating DataFrame for meriging columns
    ngramdf = pd.DataFrame(columns = df_new2.columns)

    n = 0
    int(n)
    for i in range(0,len(df_new2["Date"])):
        if(n>0):
            ngramdf.loc[df_new2.index[n]] = df_new2.iloc[n]+", "+ngramdf.iloc[n-1]
        else:
            ngramdf.loc[df_new2.index[n]] = df_new2.iloc[n]
        n=n+1
    del ngramdf["Date"]
    ngramdf.fillna("missing", inplace=True)


    #Creating a Folder name process_dir to store all the files
    process_dir=os.path.join(os.getcwd(),"dags/process_dir/")
    def create_dir():
        if not os.path.exists(process_dir):
            os.mkdir(process_dir)
    create_dir()

   
   

    #using function in dataframe
    unigramlist = []
    bigramlist = []
    trigramlist = []

    for i in ngramdf.columns:

        for j in ngramdf[i]:     
            if(j==np.nan or isinstance(j, float)):
                pass
            
            elif(len(j)==1):
                uni = n_grams(j,1)
                if(uni==[]):
                    pass
                else:
                    unigramlist.append(uni)
                    
            elif(len(j)==2):
                uni = n_grams(j,1)
                if(uni==[]):
                    pass
                else:
                    unigramlist.append(uni)
                bi = n_grams(j,2)
                if(bi==[]):
                    pass
                else:
                    bigramlist.append(bi)
                    
            elif(len(j)>2):
                uni = n_grams(j,1)
                if(uni==[]):
                    pass
                else:
                    unigramlist.append(uni)
                bi = n_grams(j,2)
                if(bi==[]):
                    pass
                else:
                    bigramlist.append(bi)
                
                tri = n_grams(j,3)
                if(tri==[]):
                    pass
                else:
                    trigramlist.append(tri)
            else:
                pass
        
        #Removing empty list
        for lst in unigramlist:
            if(lst==[]):
                unigramlist.remove(lst)
        for lst in bigramlist:
            if(lst==[]):
                bigramlist.remove(lst)            
        for lst in trigramlist:
            if(lst==[]):
                trigramlist.remove(lst) 
                
        
        #making the file
        dte = str(date.today())
        textfile1 = open(process_dir+"\\"+ i +"_" + dte +"_unigram.txt", "w")
        for element in unigramlist:
            element = str(element)
            textfile1.write(element + "\n")
        textfile1.close()
            
        textfile2 = open(process_dir+"\\"+ i + "_" + dte +"_bigram.txt", "w")
        for element in bigramlist:
            element = str(element)
            textfile2.write(element + "\n")
        textfile2.close()
        
        textfile3 = open(process_dir+"\\"+ i + "_" + dte +"_trigram.txt", "w")    
        for element in trigramlist:
            element = str(element)
            textfile3.write(element + "\n")
        textfile3.close()
        
        unigramlist = []
        bigramlist = []
        trigramlist = []


    #Connecting with minio
    from minio import Minio
    clientminio = Minio(
        endpoint="192.168.0.8:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )

    #Bucket name
    if clientminio.bucket_exists("ngram"):
        bucket_name = "ngram"
    else:
        clientminio.make_bucket("ngram")    
        bucket_name = "ngram"

    #Adding the file in minio bucket name 'ngram'
    for i, filename in enumerate(os.listdir(process_dir)):
        predict_label,dte,lst=tuple(filename.split("_"))
        dte = str(date.today())
        result = clientminio.fput_object(bucket_name, predict_label+"/"+dte+"/"+lst, os.path.join(process_dir,filename))
        print(result.object_name, result.version_id)

        # remove this file from source dir
        os.remove(process_dir+"/"+filename)

    

# ===============================================================================



# Default settings applied to all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(seconds=120)
}

dag=DAG('data_transform',
         start_date=datetime(2021, 12, 16),
         max_active_runs=2,
         schedule_interval= "@daily",
         default_args=default_args,
         catchup=False
         ) 


start_dummy = DummyOperator(
    task_id='start',
    dag=dag,
    )

connect_mongo_postgress = PythonOperator(
  task_id='connect_mongo_postgress',
  python_callable=connect_mongo_postgress, #Registered method
  provide_context=True,
  dag=dag
)


connect_minio = PythonOperator(
  task_id='connect_minio',
  python_callable=connect_minio, #Registered method
  provide_context=True,
  dag=dag
)

end_dummy = DummyOperator(
    task_id='end',
    dag=dag,
    )

start_dummy >> connect_mongo_postgress >> connect_minio >> end_dummy
