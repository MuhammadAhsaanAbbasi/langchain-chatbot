from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
from .chatbot import langChain_Chatbot
from .settings import LANGCHAIN_API_KEY
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_API_KEY"]=str(LANGCHAIN_API_KEY)
    yield


app = FastAPI(
    title="FastAPI",
    description="FastAPI",
    version="0.0.1",
    lifespan=lifespan,
    contact={
        "name": "Muhammad Ahsaan Abbasi",
        "email": "mahsaanabbasi@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    },
    root_path="/langchain-chatbot",
    root_path_in_servers=True,
    docs_url="/docs"
)

app.add_middleware(
    SessionMiddleware, secret_key="!secret")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello Langchain World!"}

@app.get('/chat')
async def langchain_chatbot(input:str):
    chat = await langChain_Chatbot(input)
    return chat