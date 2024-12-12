from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import nest_asyncio
import uvicorn
import onnxruntime
import pickle

# Enable nested asyncio loops (required for Jupyter)
nest_asyncio.apply()

# Load the model and LabelEncoder and Scaler
onnx_sess = onnxruntime.InferenceSession('bc-classification.onnx')
input_name = onnx_sess.get_inputs()[0].name
output_name = onnx_sess.get_outputs()[0].name
#print(input_name, output_name)

with open('bc-classify_le.pkl', 'rb') as f:
    le = pickle.load(f)

with open('bc-classify_sc.pkl', 'rb') as f:
    sc = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

class InputData(BaseModel):
    inputs: list  # List of input data samples


# Prepare input data
input_data = np.random.randn(1, 30).astype(np.float32)
#print(input_data)


@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.inputs)
    sca = sc.transform(input_array)
    predictions = onnx_sess.run([output_name], {input_name: sca.astype(np.float32)})[0]
    predicted_classes = le.inverse_transform(np.argmax(predictions, axis=1))
    return {"predictions": predicted_classes.tolist()}


#def predict1(ip):
#    input_array = ip
#    sca = sc.transform(input_array)
#    predictions = onnx_sess.run([output_name], {input_name: sca.astype(np.float32)})[0]
#    predicted_classes = le.inverse_transform(np.argmax(predictions, axis=1))
#    return {"predictions": predicted_classes.tolist()}

#predict1(input_data)

###################### Sample Json Input for the rest call #####################
#{
#  "inputs": [
#    [-1.9342617 ,  0.8232532 ,  2.1899416 , -0.15319526, -1.7052144 ,
#        -1.2045498 ,  0.56847507,  1.159132  ,  0.33985722,  1.378682  ,
#         1.1322894 , -0.00966804, -0.74932456,  0.7979166 ,  0.29643324,
#        -1.5207571 , -0.03744063, -1.8945767 ,  0.28786647, -0.48198155,
#         0.58975315,  0.41298273,  1.8576821 ,  0.7678462 ,  1.723527  ,
#         0.19840388,  0.31719556,  1.011919  , -1.7629695 , -0.06272263]
#  ]
#}
#################################################################################
# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8484)
