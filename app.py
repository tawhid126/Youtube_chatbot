from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
import streamlit as st
from typing import List
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# video_id = "LPZh9BOjkQs"


def extract_video_id(url: str):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    else :
        video_id = url.split('/')[-1].split('?')[0]
        return video_id


def get_youtube_transcript(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript_text = " ".join([item.text for item in transcript_list])
        return transcript_text

    except TranscriptsDisabled:
        return "we get no transcript"


def text_splitter(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([text])
    return chunks


def embedding_gen(chunks: List[Document], embeddings : GoogleGenerativeAIEmbeddings):
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


def retriever_fn(query: str, vectorstore: FAISS):
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.9}
    )
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


st.header("YouTube Chatbot with Gemini-2.5-flash")


url = st.text_input("Enter the youtube url")
query = st.text_input("Enter your question")

if st.button("Find Ans"):

    with st.spinner("Bot Is Thinking....."):

        video_id = extract_video_id(url)
        
        transcript = get_youtube_transcript(video_id)

        chunks = text_splitter(transcript)

        vectorstore = embedding_gen(chunks, embeddings)

        # Create prompt template
        prompt_template = PromptTemplate(
            template="""Use the following context to answer the question.
        If the answer is not in the context, say "I don't know".  
        context: {context}
        question: {question}
        answer:""",
            input_variables=["context", "question"],
        )

        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        parser = StrOutputParser()

        parallel_chain = RunnableParallel(
            context=RunnableLambda(lambda x: retriever_fn(x["question"], vectorstore)),
            question=RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
        )

        full_chain = parallel_chain | prompt_template | model | parser

        result = full_chain.invoke({"question": query})
        st.markdown("Conversation : ")

        st.write(f"Your question : {query}")
        st.write(f"Chatbot reply : {result}")
