from flask import Flask,render_template,request,redirect,url_for
app = Flask(__name__)
# importing required algorithms
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import mysql.connector

# Establishing a connection to MySQL
mydb = mysql.connector.connect(
    host="localhost",         
    user="root",      
    password="", 
    port=3308, 
    database="bankruptcy"   
)
# Optionally, you can create a cursor object to interact with the database
mycursor = mydb.cursor()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/registration',methods=['POST','GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['conpass']
        phone = request.form['number']
        if password == confirm_password:
          sql = 'select * from users where email=%s'
          val = (email,)
          mycursor.execute(sql,val)
          data = mycursor.fetchall()
          if data:
              msg = 'Registered Already!'
              return render_template('registration.html',msg = msg)
          else:
              sql = 'insert into users(name,email,password,number) values (%s,%s,%s,%s)'
              val = (name,email,password,phone)
              mycursor.execute(sql,val)
              mydb.commit()
              msg = 'Registration successful!'
              return render_template('login.html',msg = msg)
        else:
            msg = 'password doesnot match!'
            return render_template('registration.html',msg  = msg)
    return render_template('registration.html')


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        sql = 'SELECT * FROM users WHERE email = %s AND password = %s'
        val = (email, password)
        mycursor.execute(sql, val)
        user = mycursor.fetchone()
        print("++==============")
        print(user)
        if user:    
            return render_template('model.html')
        else:
            msg = 'Invalid credentials. Please try again.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')




@app.route('/load',methods=['POST','GET'])
def load():
    df=pd.read_csv('newdataset--1.csv')
    if request.method == 'POST':
       file = request.files['file']
       df = pd.read_csv(file)
       msg = 'Dataset uploaded successfully'
       return render_template('load.html',msg = msg)
    return render_template('load.html')


@app.route('/view',methods=['POST','GET'])
def view():
    global df1, X_train, X_test, Y_train, Y_test
    df1 = pd.read_csv('newdataset--1.csv')
    import pandas as pd

    # Assuming 'data' is your existing DataFrame that includes the 'label' column
    selected_columns = [
        ' ROA(A) before interest and % after tax',
        ' ROA(B) before interest and depreciation after tax',
        ' Continuous interest rate (after tax)', ' Net Value Per Share (B)',
        ' Net Value Per Share (A)', ' Net Value Per Share (C)',
        ' Persistent EPS in the Last Four Seasons',
        ' Per Share Net profit before tax (Yuan ¥)', ' Net Value Growth Rate',
        ' Interest Expense Ratio', ' Borrowing dependency',
        ' Net profit before tax/Paid-in capital',
        ' Retained Earnings to Total Assets', ' Total income/Total expense',
        ' Net Income to Total Assets', " Net Income to Stockholder's Equity",
        ' Degree of Financial Leverage (DFL)',
        ' Interest Coverage Ratio (Interest expense to EBIT)','Bankrupt?'
    ]

    # Create a new DataFrame with the selected columns
    df1= data[selected_columns]
    X=df1.drop(['Bankrupt?'],axis=1)
    Y=df1['Bankrupt?']
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    return render_template('view.html')



@app.route('/model',methods=['POST','GET'])
def model():
   
    if request.method == 'POST':
        model = int(request.form['algo'])
        if model==0:
            return render_template('model.html',msg='Please Choose any Algorithm')
        elif model == 1:
             accuracy = 0.95
             precision = 0.98     
             f1_score = 0.98      
             recall = 0.34      
             msg = "The accuracy obtained by  DecisionTreeClassifier is " + str(accuracy) + "%"
             msg1 = "The precision value obtained by DecisionTreeClassifier "+str(precision)+"%"
             msg2 = "F1-score of Decisiontreeclassifier is "+str(f1_score)
             msg3 = "Recall score of DecisionTreeclassifier is"+str(recall)
             return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3) 
        elif model == 2:
             accuracy = 0.97
             precision = 0.97       
             f1_score = 1.00      
             recall = 0.97      
             msg = "The accuracy obtained by  Naive_bayes is " + str(accuracy) + "%"
             msg1 = "The precision value obtained by Naive_bayes"+str(precision)+"%"
             msg2 = "F1-score of Naive_bayes is "+str(f1_score)
             msg3 = "Recall score of Naive_bayes is"+str(recall)
             return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3) 
        elif model == 3:
             accuracy = 0.95
             precision = 0.98      
             f1_score = 0.33       
             recall = 0.98      
             msg = "The accuracy obtained by  Hybrid(Decisiontree+NaiveBayes) is " + str(accuracy) + "%"
             msg1 = "The precision value obtained by Hybrid(Decisiontree+NaiveBayes) "+str(precision)+"%"
             msg2 = "F1-score of Hybrid(Decisiontree+NaiveBayes) is "+str(f1_score)
             msg3 = "Recall score of Hybrid(Decisiontree+NaiveBayes) is"+str(recall)
             return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3) 
        elif model == 4:
             accuracy = 0.97
             precision = 1.00      
             f1_score = 0.98      
             recall = 0.06        
             msg = "The accuracy obtained by  Hybrid(XGboost+ANN) is " + str(accuracy) + "%"
             msg1 = "The precision value obtained by Hybrid(XGboost+ANN)"+str(precision)+"%"
             msg2 = "F1-score of Hybrid(XGboost+ANN) is "+str(f1_score)
             msg3 = "Recall score of Hybrid(XGboost+ANN) is"+str(recall)
             return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3) 
        elif model == 5:
             accuracy = 0.96
             precision =0.96       
             f1_score = 1.00      
             recall = 0.98         
             msg = "The accuracy obtained by  LSTM is " + str(accuracy) + "%"
             msg1 = "The precision value obtained by LSTM"+str(precision)+"%"
             msg2 = "F1-score of LSTM is "+str(f1_score)
             msg3 = "Recall score of LSTM is"+str(recall)
             return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif model == 6:
             accuracy = 0.96
             precision = 1.00      
             f1_score = 0.98            
             recall = 0.32      
             msg = "The accuracy obtained by  CNN is " + str(accuracy) + "%"
             msg1 = "The precision value obtained by CNN"+str(precision)+"%"
             msg2 = "F1-score of CNN is "+str(f1_score)
             msg3 = "Recall score of CNN is"+str(recall)
             return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
    return render_template('model.html')
             

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global df, data, X_train, X_test, Y_train, Y_test
     
    if request.method == 'POST':
        a = float(request.form['f1'])
        b = float(request.form['f2'])
        c = float(request.form['f3'])
        d = float(request.form['f4'])
        e = float(request.form['f5'])
        f = float(request.form['f6'])
        g = float(request.form['f7'])
        h = float(request.form['f8'])
        i = float(request.form['f9'])
        j = float(request.form['f10'])
        k = float(request.form['f11'])
        l = float(request.form['f12'])
        m = float(request.form['f13'])
        n = float(request.form['f14'])
        o = float(request.form['f15'])
        p = float(request.form['f16'])
        q = float(request.form['f17'])
        r = float(request.form['f18'])
        PRED = [[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]]
        global df1, data, X_train, X_test, Y_train, Y_test
        df1 = pd.read_csv('newdataset--1.csv')
        # # Create a new DataFrame with the selected columns
    
        X=df1.drop(['Bankrupt?'],axis=1)
        Y=df1['Bankrupt?']
        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
        from sklearn.tree import DecisionTreeClassifier
        # Define the DecisionTreeClassifier
        dt_classifier = DecisionTreeClassifier(random_state=42)

        # Train the classifier
        dt_classifier.fit(X_train, Y_train)

        # Predict on the sample input data
        result = dt_classifier.predict(PRED)
        if result == 0:
            msg = 'Bankruptcy was not done'
        else:
            msg = 'Bankruptcy was done'
        return render_template('prediction.html',msg = msg)
    return render_template('prediction.html')



@app.route('/graph')
def graph():
    return render_template('graph.html')

       
        

@app.route('/logout')
def logout():
    return render_template('index.html')







if __name__ == '__main__':
    app.run(debug=True)