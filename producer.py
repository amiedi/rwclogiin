from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


app = FastAPI()

#Allow all origins

origins = ["*"]
app.add_middleware(

    CORSMiddleware,
    allow_origins= origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)
class Name(BaseModel):
    first_name: str
    last_name : str

@app.post('/login')

async def welcome(name:Name):
    return("Request confirmation  Ms/Ms {name.fist},{name.last_name}!")

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1' , port = 8801)



