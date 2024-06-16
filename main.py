from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

@app.get("/")
def hello_world():
    return "This is Info Chatter!"

