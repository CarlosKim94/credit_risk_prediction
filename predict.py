import pickle
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.encoders import jsonable_encoder

model_file = 'model/model_depth_25_estimator_60_0.858.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

@app.post('/predict')

async def predict(request: Request):
    client = await request.json()
    
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    reject = y_pred >= 0.5

    result = {
        'loan_default_probability': float(y_pred),
        'reject': bool(reject)
    }
    return jsonable_encoder(result)

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696, reload=True)