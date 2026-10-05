import os
import sys
import uvicorn
from fastapi import FastAPI, Form, Request
from starlette.templating import _TemplateResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.exception import CustomException
from src.logger import get_logger
logger = get_logger(__name__)
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
app =FastAPI(title = "Covid Prediction Clinic")

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("BASE_DIR:", BASE_DIR)
print("Template exists:", os.path.exists(os.path.join(BASE_DIR, "templates", "index.html")))

# Mounting CSS file
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Set template folder
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> _TemplateResponse:
    logger.info("Home page accessed...")
    return templates.TemplateResponse(request, "index.html")


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    logger.info("Predict form page accessed...")
    return templates.TemplateResponse(request, "predict.html", {"result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_result(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    fever: float = Form(...),
    cough: str = Form(...),
    city: str = Form(...)
):
    try:
        logger.info(f"Prediction request received :: age{age}, gender{gender}, fever{fever}, cough{cough}, city{city}")
        custom_data =CustomData(age=age, gender=gender,fever=fever, cough=cough, city=city)
        data_df =custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        result, probability =predict_pipeline.predict(data_df)
        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "result": result,
                "probability": probability,
                "form_data":{
                    "age":age,
                    "gender":gender,
                    "fever":fever,
                    "cough":cough,
                    "city":city
                }
            }

        )
        
    except Exception as e:
        raise CustomException(e,sys)

@app.get("/health")
async def health_check() ->dict[str, str]:
    logger.info("Monitering alert....")
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)