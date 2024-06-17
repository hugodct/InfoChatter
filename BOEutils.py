import PyPDF2
import re, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name="text-embedding-3-small")

def extract_anexos(infile_path, outfile_path):
    # Detect page range for each anexo
    regex = re.compile(r"ANEXO [I,V,X]*")
    subtitle_regex = re.compile('\\n(.*)\\n')
    reader = PyPDF2.PdfReader(infile_path)
    anexos = []
    pagespan = []
    pagespan_list = []

    for count, page in enumerate(reader.pages):

        pagespan.append(count)
        text = page.extract_text()
        match = re.search(regex, text)

        if match:
            anexo = {}
            end_position = match.span()[1]      # Get end position of ANEXO XXX
            subtitle_extract = text[end_position: end_position + 300]     # Extract text where the title is
            subtitle_match = re.search(subtitle_regex, subtitle_extract)

            anexo['name'] = match.group(0)
            anexo['subtitle'] = subtitle_match.group(0).strip()
            anexos.append(anexo)
            pagespan_list.append(pagespan)
            pagespan = []

    pagespan_list.append(pagespan)
    pagespan_list.pop(0)

    # Append page ranges to anexos
    for count, item in enumerate(anexos):
        item['pages'] = pagespan_list[count]

    # Extract files
    for count, anexo in enumerate(anexos):
        writer = PyPDF2.PdfWriter()

        for page in anexo['pages']:
            writer.add_page(reader.pages[page - 1])
            last_page = page
        writer.add_page(reader.pages[last_page])    # Write last page

        with open(outfile_path + "/anexo" + str(count) + ".pdf", "wb") as out:
            writer.write(out)


def vectorize_pdf(ChromaClient, collectionhash, pdffile, pdfname):
    # Splits and embeds pdf in a collection of the same name
    loader = PyPDFLoader(pdffile)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    Chroma.from_documents(all_splits, OpenAIEmbeddings(model="text-embedding-3-small"), collection_name=collectionhash, client=ChromaClient, collection_metadata={"filename": pdfname})


def extract_info(ChromaClient, collectionhash, info_to_extract: str):
    # Load from disk
    collection = Chroma(client=ChromaClient, collection_name=collectionhash, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

    # Retriever
    retriever = collection.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate.from_template("Eres un asistente que responde a preguntas. Utiliza los "
                                          "siguientes elementos del contexto recuperado para responder a la pregunta. "
                                          "Si no conoces la respuesta, indica simplemente que no la conoces. Utiliza "
                                          "seis frases como máximo Sé  claro y responde de la mejor manera posible a la pregunta. "
                                          "\nPregunta: {question} "
                                          "\nContexto: {context}."
                                          "\nRespuesta:")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain.invoke(info_to_extract)

def vectorize_web(ChromaClient, collectionhash, weburl, webname):
    loader = WebBaseLoader(weburl)
    data = loader.load()
