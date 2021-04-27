# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import numpy as np
import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)

@app.route("/", methods=["GET"])
def general():
	return "Telecom dataset churn prediction process"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		request_json = flask.request.get_json()
		SeniorCitizen = request_json["SeniorCitizen"]
		tenure = request_json['tenure']
		MonthlyCharges = request_json['MonthlyCharges']
		gender_Male = request_json['gender_Male']
		Partner_Yes = request_json['Partner_Yes']
		Dependents_Yes = request_json['Dependents_Yes']
		PhoneService_Yes = request_json['PhoneService_Yes']
		MultipleLines_No = request_json['MultipleLines_No phone service']
		MultipleLines_Yes = request_json['MultipleLines_Yes']
		InternetService_Fiber = request_json['InternetService_Fiber optic']
		InternetService_No = request_json['InternetService_No']
		OnlineSecurity_No = request_json['OnlineSecurity_No internet service']
		OnlineSecurity_Yes = request_json['OnlineSecurity_Yes']
		OnlineBackup_No = request_json['OnlineBackup_No internet service']
		OnlineBackup_Yes = request_json['OnlineBackup_Yes']
		DeviceProtection_No = request_json['DeviceProtection_No internet service']
		DeviceProtection_Yes = request_json['DeviceProtection_Yes']
		TechSupport_No = request_json['TechSupport_No internet service']
		TechSupport_Yes = request_json['TechSupport_Yes']
		StreamingTV_No = request_json['StreamingTV_No internet service']
		StreamingTV_Yes = request_json['StreamingTV_Yes']
		StreamingMovies_No = request_json['StreamingMovies_No internet service']
		StreamingMovies_Yes = request_json['StreamingMovies_Yes']
		Contract_One = request_json['Contract_One year']
		Contract_Two = request_json['Contract_Two year']
		PaperlessBilling_Yes = request_json['PaperlessBilling_Yes']
		PaymentMethod_Credit = request_json['PaymentMethod_Credit card (automatic)']
		PaymentMethod_Electronic = request_json['PaymentMethod_Electronic check']
		PaymentMethod_Mailed = request_json['PaymentMethod_Mailed check']
		preds = model.predict_proba(pd.DataFrame({"SeniorCitizen": [SeniorCitizen],
												  "tenure": [tenure],
												  "MonthlyCharges": [MonthlyCharges],
												  'gender_Male': [gender_Male],
												  'Partner_Yes': Partner_Yes,
												  'Dependents_Yes': Dependents_Yes,
												  'PhoneService_Yes': PhoneService_Yes,
												  'MultipleLines_No phone service': MultipleLines_No,
												  'MultipleLines_Yes': MultipleLines_Yes,
												  'InternetService_Fiber optic': InternetService_Fiber,
												  'InternetService_No': InternetService_No,
												  'OnlineSecurity_No internet service': OnlineSecurity_No,
												  'OnlineSecurity_Yes': OnlineSecurity_Yes,
												  'OnlineBackup_No internet service': OnlineBackup_No,
												  'OnlineBackup_Yes': OnlineBackup_Yes,
												  'DeviceProtection_No internet service': DeviceProtection_No,
												  'DeviceProtection_Yes': DeviceProtection_Yes,
												  'TechSupport_No internet service': TechSupport_No,
												  'TechSupport_Yes': TechSupport_Yes,
												  'StreamingTV_No internet service': StreamingTV_No,
												  'StreamingTV_Yes': StreamingTV_Yes,
												  'StreamingMovies_No internet service': StreamingMovies_No,
												  'StreamingMovies_Yes': StreamingMovies_Yes,
												  'Contract_One year': Contract_One,
												  'Contract_Two year': Contract_Two,
												  'PaperlessBilling_Yes': PaperlessBilling_Yes,
												  'PaymentMethod_Credit card (automatic)': PaymentMethod_Credit,
												  'PaymentMethod_Electronic check': PaymentMethod_Electronic,
												  'PaymentMethod_Mailed check': PaymentMethod_Mailed}))
		data["predictions"] = preds[:, 1][0]
		data["description"] = "123"
		# indicate that the request was a success
		data["success"] = True
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "./models/rfc_pipeline.dill"
	load_model(modelpath)
	app.run()