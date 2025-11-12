import requests

client = { "person_age": 27,
    "person_income": 59000,
    "person_home_ownership": "RENT",
    "person_emp_length": 2.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "loan_amnt": 35000,
    "lloan_int_rate": 16.02,
    "loan_percent_income": 0.59,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 3}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=client).json()
print(response)
if response['reject'] == True:
    print("Reject this loan application...")