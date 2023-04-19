import os
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
import gradio as gr


class customLLM(LLM):
    # model_name = "microsoft/DialoGPT-medium"
    model_name = "google/flan-t5-small"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


class QnABot:
    def __init__(self, dbpath='metamorphosis_index', model_name="google/flan-t5-small") -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.db = FAISS.load_local(dbpath, self.embeddings)
        self.template = """
        for the following context start answering questions, if you don't find answer in the context just say that "you don't know the answer"
        CONTEXT: {context}
        QUESTION: {question} 
        """

    def qna_bot(self, input_text):
        context = ' '.join([doc.page_content for doc in self.db.similarity_search(input_text)[:2]])
        prompt = PromptTemplate(input_variables=['context', 'question'], template=self.template)
        qna_chain = LLMChain(llm=customLLM(), prompt=prompt)
        response = qna_chain.run({'context': context, 'question': input_text})
        return response


if __name__ == '__main__':
    bot = QnABot()
    iface = gr.Interface(fn=bot.qna_bot,
                     inputs=gr.inputs.Textbox(lines=4, label="Enter your text"),
                     outputs="text",
                     title="Question Answering Bot")

    iface.launch(share=True)