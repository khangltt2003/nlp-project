from fastapi import FastAPI
from inverted_index import Inverted_Index
from contextlib import asynccontextmanager
import json
inverted_index = Inverted_Index()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    inverted_index.load()
    yield
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root ():
  return {"example":"hello world"}

@app.get("/search")
async def search(query: str):
  terms = query.split(" ")
  res = inverted_index.method2(terms[0], terms[1], terms[2])
  return  { "result" : res}