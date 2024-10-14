from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

@app.get('/')
def route_default():
    return 'Welcome to API'

@app.get('/database_connection')
def db_connection():
    return 'Interface de Conexão com Database'
