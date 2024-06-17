import chromadb
from fastapi import FastAPI, Request, HTTPException, UploadFile
import aiofiles
from urllib.parse import urlparse
from BOEutils import *

if os.name == 'nt':
    os.environ["TEMPFIlE_PATH"] = os.path.join("vectordb", "tempfile.pdf")
    os.environ["PDFDICT_PATH"] = os.path.join('vectordb', 'pdf_dict.pkl')
else:
    os.environ["TEMPFIlE_PATH"] = os.path.join(os.sep, 'storage', 'InfoChatter', "vectordb", "tempfile.pdf")
    os.environ["PDFDICT_PATH"] = os.path.join(os.sep, 'storage', 'InfoChatter', 'vectordb', 'pdf_dict.pkl')
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


app = FastAPI()
ChromaClient = chromadb.PersistentClient(path=os.path.join(os.sep, 'storage', 'InfoChatter', "vectordb"))

@app.get("/")
def hello_world():
    return "This is Info Chatter!"

@app.get("/vectorize_pdf")
async def vectorizepdf(in_file: UploadFile, pdfhash: str):
    # Vectorizes PDF content and stores it in collection with name pdfhash
    async with aiofiles.open(os.environ.get('TEMPFIlE_PATH'), 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write

    vectorize_pdf(ChromaClient, pdfhash, os.environ.get('TEMPFIlE_PATH'), in_file.filename)

    # Save pdf name and hash in dict
    dictpath = os.environ.get('PDFDICT_PATH')
    save_to_hashdict(dictpath, in_file.filename, pdfhash)

    os.remove(os.environ.get('TEMPFIlE_PATH'))
    return "PDF vectorized succesfully, saved at collection " + pdfhash

@app.get("/vectorize_web")
async def vectorizeweb(request: Request, webhash: str):
    body = await request.json()
    weburl = body["weburl"]

    webname = urlparse(weburl).netloc
    vectorize_web(ChromaClient, webhash, weburl, webname)

    dictpath = os.environ.get('PDFDICT_PATH')
    save_to_hashdict(dictpath, webname, webhash)

    # Save to pdf dict
    return "Web vectorized succesfully, saved at collection " + webhash

@app.get("/ask_info")
async def extractinfo(request: Request, collectionhash: str):
    # Uses collection in pdfhash to answer question in request
    body = await request.json()
    question = body["question"]
    return extract_info(ChromaClient, collectionhash, question)


