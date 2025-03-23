from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from utils import *
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "processed_data.csv"


# Add Pydantic models for request validation
class ProcessRequest(BaseModel):
    file_path: str

class QuestionRequest(BaseModel):
    question: str

# Modify the endpoints to use these models
@app.post("/process")
async def process_data(request: ProcessRequest = Body(...)):
    """Endpoint to preprocess and store data"""
    try:
        df = pd.read_csv(request.file_path)
        processed_df = preprocess_hotel_data(request.file_path)
        processed_df.to_csv(DATA_PATH, index=False)
        return {"status": "success", "message": "Data processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest = Body(...)):
    """Endpoint for Q&A"""
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=400, detail="Process data first")
            
        df = pd.read_csv(DATA_PATH)
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        response = rag_hotel_qa(df, request.question)
        return {
            "status": "success",
            "question": response['question'],
            "answer": response['answer'],
            "context": response['context']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analytics")
async def get_analytics():
    """Endpoint to generate analytics reports"""
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=400, detail="Process data first")
            
        df = pd.read_csv(DATA_PATH)
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        results = analyze_hotel_data(df)
        
        # Convert plots to base64 for API response
        import base64
        from io import BytesIO
        
        analytics_data = {}
        for key in results:
            if 'plot' in results[key]:
                buf = BytesIO()
                results[key]['plot'].savefig(buf, format='png')
                buf.seek(0)
                analytics_data[key] = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            "status": "success",
            "analytics": analytics_data,
            "stats": {
                "cancellation_rate": results['cancellation_analysis']['rate'],
                "top_countries": results['geographical_distribution']['data'].to_dict()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)