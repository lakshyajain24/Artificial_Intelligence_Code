from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from pymongo import MongoClient
from random import randint
from datetime import date
from datetime import datetime
import pandas as pd
import os

mydir=os.path.join(os.getcwd(),"dags/")

#f = open("dags/count.txt", "r")
#count = f.read()
count=int(Variable.get('my_iterator'))

df = pd.read_csv(mydir+"answers.csv",encoding = "ISO-8859-1")


# ================================custom functions=============================

def myfunction():
    lines = [str(Variable.get('my_iterator_date'))]
    for i in lines:
        if str(date.today()) not in i:
            global count
            _id = []
            body = []
            question = []
            ct=[]
            ut=[]
            ct1=[]
            ut1=[]
            date1 = list(pd.date_range("2021-04-02",periods=200))
            #date2 = list(pd.date_range("2021-12-22",periods=200))
            for i in range(0,len(df["updatedAt"])):
                temp =df["updatedAt"][i].split("T")[0]
                
                
                if(temp==str(date1[count]).split(" ")[0]):
                    
                    _id.append(df['_id'][i])
                    body.append(df['Body.'][i])
                    question.append(df["question"][i])
                    ct.append(df["createdAt"][i].split("T")[1])
                    ut.append(df["updatedAt"][i].split("T")[1])
                    #ct1.append(str(date2[count]).split(" ")[0])
                    #ut1.append(str(date2[count]).split(" ")[0])
            
            
            #client = MongoClient(port=27017)
            client = MongoClient( host= '192.168.0.8', port=27017,
                          username='root',
                         password='rootpassword')
            db=client.answers
            for x in range(len(_id)):
                business = {
                    '_id' : _id[x],
                    'Body.' : body[x],
                    #'createdAt' : ct1[x]+"T"+ct[x],
                    'createdAt' : str(date.today())+"T"+ct[x],
                    'question' : question[x],
                    #'updatedAt' : ut1[x]+"T"+ut[x]
                    'updatedAt' : str(date.today())+"T"+ut[x]
                    }
                result=db.data.insert_one(business)
            count+=1
"""
            with open(mydir+'count.txt','w') as file1:
                file1.write(str(count))
            with open(mydir+'date.txt','w') as file2:
                    file2.write(str(date.today()))

"""
# ===============================================================================
def setup_var_and_xcom(**kwargs):
    ti = kwargs['ti']

    date_iterator = str(Variable.get('my_iterator_date', default_var=0))
    n = datetime.strptime(date_iterator, '%Y-%m-%d')
    final = str(n + timedelta(days=1))
    final = final.split(" ")
    date_iterator = final[0]
    Variable.set('my_iterator_date', date_iterator)

    iterator = int(Variable.get('my_iterator', default_var=0))
    iterator += 1
    Variable.set('my_iterator', iterator)


# Default settings applied to all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(seconds=120)
}

dag=DAG('survey',
         start_date=datetime(2021, 12, 22),
         max_active_runs=2,
         schedule_interval='@daily',
         default_args=default_args,
         catchup=False
         ) 

start_ = DummyOperator(
        task_id='start',
        dag=dag,
        )

# Step 4 - Create a Branching task to Register the method in step 3 to the branching API


myfunction = PythonOperator(
  task_id='myfunction',
  python_callable=myfunction, #Registered method
  provide_context=True,
  dag=dag
)

setup_var_and_xcom = PythonOperator(
    task_id='setup_var_and_xcom',
    python_callable=setup_var_and_xcom,
    provide_context=True,
    dag=dag
)


end_dummy = DummyOperator(
    task_id='end',
    dag=dag,
    )

start_ >> myfunction>> setup_var_and_xcom >>end_dummy
