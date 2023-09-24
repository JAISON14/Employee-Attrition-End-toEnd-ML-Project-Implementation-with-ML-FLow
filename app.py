from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from Employee_Attition_End_to_end_ML_project_with_MLflow.pipeline.prediction import PredictionPipeline



app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET', 'POST'])  # route to display the home page
def homePage():
    return render_template("index.html")

    
    # Handling new input variables from the updated HTML form
    if request.method == 'POST':
        age = request.form['Age']
        businesstravel = request.form['BusinessTravel']
        dailyrate = request.form['DailyRate']
        department = request.form['Department']
        distancefromhome = request.form['DistanceFromHome']
        education = request.form['Education']
        educationfield = request.form['EducationField']
        employeecount = request.form['EmployeeCount']
        employeenumber = request.form['EmployeeNumber']
        environmentsatisfaction = request.form['EnvironmentSatisfaction']
        gender = request.form['Gender']
        hourlyrate = request.form['HourlyRate']
        jobinvolvement = request.form['JobInvolvement']
        joblevel = request.form['JobLevel']
        jobrole = request.form['JobRole']
        jobsatisfaction = request.form['JobSatisfaction']
        maritalstatus = request.form['MaritalStatus']
        monthlyincome = request.form['MonthlyIncome']
        monthlyrate = request.form['MonthlyRate']
        numcompaniesworked = request.form['NumCompaniesWorked']
        over18 = request.form['Over18']
        overtime = request.form['OverTime']
        percentsalaryhike = request.form['PercentSalaryHike']
        performancerating = request.form['PerformanceRating']
        relationshipsatisfaction = request.form['RelationshipSatisfaction']
        standardhours = request.form['StandardHours']
        stockoptionlevel = request.form['StockOptionLevel']
        totalworkingyears = request.form['TotalWorkingYears']
        trainingtimeslastyear = request.form['TrainingTimesLastYear']
        worklifebalance = request.form['WorkLifeBalance']
        yearsatcompany = request.form['YearsAtCompany']
        yearsincurrentrole = request.form['YearsInCurrentRole']
        yearssincelastpromotion = request.form['YearsSinceLastPromotion']
        yearswithcurrmanager = request.form['YearsWithCurrManager']

        # Add your code to use these variables here



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            age = int(request.form['Age'])
            businesstravel = request.form['BusinessTravel']
            dailyrate = int(request.form['DailyRate'])
            department = request.form['Department']
            distancefromhome = int(request.form['DistanceFromHome'])
            education = int(request.form['Education'])
            educationfield = request.form['EducationField']
            employeecount = int(request.form['EmployeeCount'])
            employeenumber = int(request.form['EmployeeNumber'])
            environmentsatisfaction = int(request.form['EnvironmentSatisfaction'])
            gender = request.form['Gender']
            hourlyrate = int(request.form['HourlyRate'])
            jobinvolvement = int(request.form['JobInvolvement'])
            joblevel = int(request.form['JobLevel'])
            jobrole = request.form['JobRole']
            jobsatisfaction = int(request.form['JobSatisfaction'])
            maritalstatus = request.form['MaritalStatus']
            monthlyincome = int(request.form['MonthlyIncome'])
            monthlyrate = int(request.form['MonthlyRate'])
            numcompaniesworked = int(request.form['NumCompaniesWorked'])
            over18 = request.form['Over18']
            overtime = request.form['OverTime']
            percentsalaryhike = int(request.form['PercentSalaryHike'])
            performancerating = int(request.form['PerformanceRating'])
            relationshipsatisfaction = int(request.form['RelationshipSatisfaction'])
            standardhours = int(request.form['StandardHours'])
            stockoptionlevel = int(request.form['StockOptionLevel'])
            totalworkingyears = int(request.form['TotalWorkingYears'])
            trainingtimeslastyear = int(request.form['TrainingTimesLastYear'])
            worklifebalance = int(request.form['WorkLifeBalance'])
            yearsatcompany = int(request.form['YearsAtCompany'])
            yearsincurrentrole = int(request.form['YearsInCurrentRole'])
            yearssincelastpromotion = int(request.form['YearsSinceLastPromotion'])
            yearswithcurrmanager = int(request.form['YearsWithCurrManager'])
       
         
            data = [
                age, businesstravel, dailyrate, department, distancefromhome,
                education, educationfield, employeecount, employeenumber,
                environmentsatisfaction, gender, hourlyrate, jobinvolvement,
                joblevel, jobrole, jobsatisfaction, maritalstatus, monthlyincome,
                monthlyrate, numcompaniesworked, over18, overtime,
                percentsalaryhike, performancerating, relationshipsatisfaction,
                standardhours, stockoptionlevel, totalworkingyears, trainingtimeslastyear,
                worklifebalance, yearsatcompany, yearsincurrentrole,
                yearssincelastpromotion, yearswithcurrmanager
            ]
            #data = np.array(data).reshape(1, 34)
            print(data)
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)