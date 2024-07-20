import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = model_path = r"C:\Users\nehar\My_All_Projects\AI-ML_bootcamp\ML_models\random_forest_model.pkl"

model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")
    expected_columns = []

def main():
    # Set the title of the web app
    st.title('Employee Attrition Prediction')

    # Add a description
    st.write('Enter employee information to predict attrited or not.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Employee Information')

        # Add input fields for features
        employee_name = st.text_input('Employee Name')
        Gender = st.selectbox("Employee's Gender", ['Female', 'Male'])
        Age = st.slider("Employee's Age", 15, 65, 30)
        BusinessTravel = st.selectbox("Business travel for the employee", ['Rarely', 'Frequently', 'Non-Travel'])
        Department = st.selectbox("The department the employee works i", ['Research & Development', 'Sales', 'Human Resources'])
        EmployeeNumber = st.slider("Employee Number", 1, 2000, 1000)
        DailyRate = st.slider("The daily rate of pay for the employee", 100, 2000, 1000)
        DistanceFromHome = st.slider(" The distance from home in miles for the employee", 0, 30, 8)
        HourlyRate = st.slider("The hourly rate of pay for the employee", 5, 100, 50)
        JobInvolvement = st.slider("The level of involvement required for the employee's job", 1, 4, 2)
        JobLevel = st.slider("The job level of the employee", 1, 4, 2)
        EducationField = st.selectbox("The field of study for the employee's education", ['Medical', 'Technical Degree', 'Life Sciences', 'Marketing', 'Other'])
        Education = st.slider("The level of education achieved by the employee", 1, 5, 2)
        EnvironmentSatisfaction = st.slider("The employee's satisfaction with their work environment.", 1, 4, 2)
        JobSatisfaction = st.slider("The employee's satisfaction with their job", 1, 4, 2)
        JobRole = st.selectbox("The role of the employee in the organizatione", ['Sales Executive', 'Manufacturing Director', 'Research Scientist', 'Sales Representative', 'Laboratory Technician', 'Healthcare Representative', 'Research Director', 'Manager', 'Human Resources'])
        MaritalStatus = st.selectbox("The marital status of the employee", ['Single', 'Married', 'Divorce'])
        MonthlyIncome = st.slider("The monthly income of the employee", 1000, 25000, 6000)
        MonthlyRate = st.slider(" The monthly rate of pay for the employee", 1000, 30000, 4000)
        NumCompaniesWorked = st.slider("The number of companies the employee has worked for", 0, 12, 3)
        OverTime = st.selectbox("Whether or not the employee works overtime", ['No', 'Yes'])
        PercentSalaryHike = st.slider("The percentage of salary hike for the employee", 1, 30, 9)
        PerformanceRating = st.slider("The performance rating of the employee", 1, 5, 4)
        RelationshipSatisfaction = st.slider("The employee's satisfaction with their relationships", 1, 4, 3)
        StockOptionLevel = st.slider("Stock Option Level of the employee", 0, 3, 2)
        TotalWorkingYears = st.slider("The total number of years the employee has worked", 1, 40, 15)
        TrainingTimesLastYear = st.slider("Number of Times Trained Last Year", 0, 9, 2)
        WorkLifeBalance = st.slider(" The employee's perception of their work-life balance", 1, 4, 2)
        YearsAtCompany = st.slider("The number of years the employee has been with the company", 1, 50, 15)
        YearsInCurrentRole = st.slider(' The number of years the employee has been in their current rol', 1, 18, 5)
        YearsSinceLastPromotion = st.slider('The number of Years Since Last Promotion', 0, 20, 6)
        YearsWithCurrManager = st.slider('The number of years the employee has been with their current managerr', 0, 20, 8)
        untrained_column = st.text_input('Additional Information (not used in prediction)')

    # Convert categorical inputs to numerical
    Gender = 1 if Gender == 'Female' else 0
    BusinessTravel = {'Non-Travel': 0, 'Rarely': 1, 'Frequently': 2}.get(BusinessTravel, 0)
    Department = {'Research & Development': 1, 'Sales': 2, 'Human Resources': 0}.get(Department, 0)
    EducationField = {'Medical': 0, 'Technical Degree': 1, 'Life Sciences': 2, 'Marketing': 3, 'Other': 4}.get(EducationField, 0)
    JobRole = {'Sales Executive': 0, 'Manufacturing Director': 1, 'Research Scientist': 2, 'Sales Representative': 3, 'Laboratory Technician': 4, 'Healthcare Representative': 5, 'Research Director': 6, 'Manager': 7, 'Human Resources': 8}.get(JobRole, 0)
    MaritalStatus = {'Single': 0, 'Married': 1, 'Divorce': 2}.get(MaritalStatus, 0)
    OverTime = 1 if OverTime == 'Yes' else 0

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'BusinessTravel': [BusinessTravel],
        'Department': [Department],
        'EmployeeNumber': [EmployeeNumber],
        'DailyRate': [DailyRate],
        'DistanceFromHome': [DistanceFromHome],
        'HourlyRate': [HourlyRate],
        'JobInvolvement': [JobInvolvement],
        'JobLevel': [JobLevel],
        'EducationField': [EducationField],
        'Education': [Education],
        'EnvironmentSatisfaction': [EnvironmentSatisfaction],
        'JobSatisfaction': [JobSatisfaction],
        'JobRole': [JobRole],
        'MaritalStatus': [MaritalStatus],
        'MonthlyIncome': [MonthlyIncome],
        'MonthlyRate': [MonthlyRate],
        'NumCompaniesWorked': [NumCompaniesWorked],
        'OverTime': [OverTime],
        'PercentSalaryHike': [PercentSalaryHike],
        'PerformanceRating': [PerformanceRating],
        'RelationshipSatisfaction': [RelationshipSatisfaction],
        'StockOptionLevel': [StockOptionLevel],
        'TotalWorkingYears': [TotalWorkingYears],
        'TrainingTimesLastYear': [TrainingTimesLastYear],
        'WorkLifeBalance': [WorkLifeBalance],
        'YearsAtCompany': [YearsAtCompany],
        'YearsInCurrentRole': [YearsInCurrentRole],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
        'YearsWithCurrManager': [YearsWithCurrManager]
    })

    # Ensure columns are in the same order as during model training
    if len(expected_columns) > 0:
        input_data = input_data.reindex(columns=expected_columns)

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]

                st.write(f'Prediction for {employee_name}: {"Attrited" if prediction[0] == 1 else "Not Attrited"}')
                st.write(f'Probability of Attrition: {probability:.2f}')

                # Plotting
                fig, axes = plt.subplots(3, 1, figsize=(8, 16))

                # Plot Attrition probability
                sns.barplot(x=['Not Attrited', 'Attrited'], y=[1 - probability, probability], ax=axes[0], palette=['blue', 'red'])
                axes[0].set_title('Attrition Probability')
                axes[0].set_ylabel('Probability')

                # Plot Probability distribution (mocked, as we don't have the actual distribution)
                sns.histplot([probability], kde=True, ax=axes[1])
                axes[1].set_title('Probability Distribution')

                # Plot Attrition pie chart
                axes[2].pie([1 - probability, probability], labels=['Not Attrited', 'Attrited'], autopct='%1.1f%%', colors=['blue', 'red'])
                axes[2].set_title('Attrition Pie Chart')

                # Display the plots
                st.pyplot(fig)

                # Provide recommendations
                if prediction[0] == 1:
                    st.success(f"{employee_name} is likely to attrite. Consider reviewing engagement strategies.")
                else:
                    st.success(f"{employee_name} is likely to stay. Continue supporting their career growth.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()





