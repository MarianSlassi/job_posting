import uvicorn
from fastapi import FastAPI, HTTPException


app = FastAPI()
if __name__ == '__main__':
    uvicorn.run("src.main_test:app", host="127.0.0.1", port=8000, reload=True)