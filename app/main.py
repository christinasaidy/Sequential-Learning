from fastapi import FastAPI
from app.presentation.api import router

app = FastAPI()
app.include_router(router)