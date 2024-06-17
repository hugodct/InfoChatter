from fastapi import FastAPI, Request, HTTPException, UploadFile
import aiofiles
import pickle
import os
from BOEutils import *

if os.name == 'nt':
    os.environ["TEMPFIlE_PATH"] = os.path.join("vectordb", "tempfile.pdf")
    os.environ["PDFDICT_PATH"] = os.path.join('vectordb', 'pdf_dict.pkl')
else:
    os.environ["TEMPFIlE_PATH"] = os.path.join(os.sep, 'storage', 'InfoChatter', "vectordb", "tempfile.pdf")
    os.environ["PDFDICT_PATH"] = os.path.join(os.sep, 'storage', 'InfoChatter', 'vectordb', 'pdf_dict.pkl')


app = FastAPI()
ChromaClient = chromadb.PersistentClient(path=os.path.join(os.sep, 'storage', 'InfoChatter', "vectordb"))

@app.get("/")
def hello_world():
    return "This is Info Chatter!"

@app.get("/vectorize_pdf")
async def vectorize_pdf(in_file: UploadFile, pdfhash: str):
    async with aiofiles.open(os.environ.get('TEMPFIlE_PATH'), 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write

    boe_object = BOE(vdbcollection=pdfhash)
    boe_object.vectorize_pdf(ChromaClient, os.environ.get('TEMPFIlE_PATH'), in_file.filename)

    # Save pdf name and hash in dict
    if not os.path.isfile(os.environ.get('TEMPFIlE_PATH')):
        with open('vectordb/pdf_dict.pkl', 'wb') as f:
            dictionary = {"key":"value"}
            pickle.dump(dictionary, f)

    with open(os.environ.get('PDFDICT_PATH'), 'rb') as f:
        loaded_dict = pickle.load(f)

    with open(os.environ.get('PDFDICT_PATH'), 'wb') as f:
        loaded_dict[in_file.filename] = pdfhash
        pickle.dump(loaded_dict, f)

    os.remove(os.environ.get('PDFDICT_PATH'))

@app.get("/ask_info")
async def extract_info(request: Request, pdfhash: str):
    boe_object = BOE(pdfhash)
    body = await request.json()
    question = body["question"]
    return boe_object.extract_info(ChromaClient, question)


