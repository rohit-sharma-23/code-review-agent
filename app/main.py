"""Main application entry point."""

from fastapi import FastAPI

app = FastAPI(title="Code Review Agent API")


@app.get("/")
def read_root():
    return {"message": "Code Review Agent API is running"}
