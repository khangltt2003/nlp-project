from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from inverted_index import Inverted_Index
from contextlib import asynccontextmanager


inverted_index = Inverted_Index()

@asynccontextmanager
async def lifespan(app: FastAPI):
    inverted_index.load()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(    
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


@app.get("/")
async def root ():
  return {"example":"hello world"}

@app.get("/search")
async def search(q: str, page: int = Query(1), limit: int = Query(10)):
  terms = q.split(" ")
  res = list(inverted_index.method2(terms[0], terms[1], terms[2]))
  
  start  = (page - 1) * limit
  end  = start + 10
  res = res[start: end]
  return  { "result" : res}