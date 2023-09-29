import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GPT4All
import ai_path as pa
def main():
    st.set_page_config(page_title="ask pdf")
    st.header("Ask anything about your pdf")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            )
        chunks = text_splitter.split_text(text)
        
        instruct_embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            #change to gpu or cpu based on FAISS using cpu or gpu
            # model_kwargs={"device":"cuda"}
            model_kwargs={"device":"cpu"}
        ) 

        knowledge_base = FAISS.from_texts(chunks,instruct_embeddings)

        user_question = st.text_input("Ask a question about the contents of your pdf")
        llm = GPT4All(
            model = pa.MODEL_PATH,
            max_tokens=2048,
                )

        if user_question:
            documents = knowledge_base.similarity_search(user_question)
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents=documents, question=user_question)
            st.write(response)

if __name__ == "__main__":
    main()