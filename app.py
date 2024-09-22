from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        gender = request.form['gender']
        race_ethnicity = request.form['race_ethnicity']
        parental_level_of_education = request.form['parental_level_of_education']
        lunch = request.form['lunch']
        test_preparation_course = request.form['test_preparation_course']
        reading_score = int(request.form['reading_score'])
        writing_score = int(request.form['writing_score'])

        # Create CustomData instance
        custom_data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # Convert to DataFrame and make prediction
        data = custom_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data)

        # Render index page with prediction
        return render_template('index.html', prediction=prediction[0])
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
