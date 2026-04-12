from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        data = CustomData(
            gender=str(request.form.get("gender")),
            race_ethnicity=str(request.form.get("ethnicity")),
            parental_level_of_education=str(request.form.get("parental_level_of_education")),
            lunch=str(request.form.get("lunch")),
            test_preparation_course=str(request.form.get("test_preparation_course")),
            reading_score=float(request.form.get("reading_score", 0)), 
            writing_score=float(request.form.get("writing_score", 0)),  
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before prediction")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("After prediction")
        return render_template("home.html", results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")