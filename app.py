from flask import Flask, render_template, request
import os
import pandas as pd
from utils.ai_analysis import classify_transactions, generate_suggestions, load_model, forecast_expenses
from utils.visuals import create_charts

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            classified_df = classify_transactions(df, model)
            suggestions, category_summary = generate_suggestions(classified_df)
            charts = create_charts(classified_df)
            forecast = forecast_expenses(classified_df)
            return render_template('result.html', tables=[classified_df.to_html(classes='data')],
                                   suggestions=suggestions, charts=charts, forecast=forecast.to_dict())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)