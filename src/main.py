from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uvicorn
from predict import predict_audio

app = FastAPI(title="Baby Cry Detection API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload folder exists
os.makedirs("uploads", exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict using your existing predict.py function
        top_label, results = predict_audio(file_path)
        return {"top_reason": top_label, "results": results}
    except Exception as e:
        return {"error": str(e)}

@app.post("/contact")
async def contact(data: dict):
    try:
        # Example: just print and return success
        print("Received contact form:", data)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/profile")
async def save_profile(data: dict):
    try:
        print("Received profile data:", data)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
