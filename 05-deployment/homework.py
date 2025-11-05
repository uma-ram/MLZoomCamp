

# code in bin File
# categorical = ['lead_source']
# numeric = ['number_of_courses_viewed', 'annual_income']

# df[categorical] = df[categorical].fillna('NA')
# df[numeric] = df[numeric].fillna(0)

# train_dict = df[categorical + numeric].to_dict(orient='records')

# pipeline = make_pipeline(
#     DictVectorizer(),
#     LogisticRegression(solver='liblinear')
# )

# pipeline.fit(train_dict, y_train)

import pickle

import uvicorn
from fastapi import FastAPI

from typing import Dict, Any

app = FastAPI(title='Course_Prediction')


filename ='pipeline_v1.bin'
with open(filename,'rb') as f_in:
    pipeline = pickle.load(f_in)

# Question - 3
input_data =  {
    'lead_source': 'paid_ads',
    'number_of_courses_viewed': 2,
    'annual_income': 79276.0
}

def predict_single(input_data):
    result = pipeline.predict_proba(input_data)[0, 1]
    return float(result)

def predict1(input_data):
    print(predict_single(input_data))

#Question - 4
input_data = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 20304.0
}

@app.post('/predict')
def predict(input_data: Dict[str, Any]):
    # print(predict_single(input_data))
    return predict_single(input_data)


if __name__ == "__main__":
    # predict1(input_data)
    uvicorn.run(app, host="0.0.0.0", port=9696)