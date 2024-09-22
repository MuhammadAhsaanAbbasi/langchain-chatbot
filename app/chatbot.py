from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from .settings import HUGGINGFACE_TOKEN
from dotenv import load_dotenv
import os

_ = load_dotenv()

llm2 = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

llm1 = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=100,
    do_sample=True,
    huggingfacehub_api_token=HUGGINGFACE_TOKEN
)


async def langChain_Chatbot(prompt: str) -> str:
    output_parser = StrOutputParser()

    # First LLM Chain: Writing a LinkedIn Post
    template1 = "You are a professional content writer. Write me an interactive LinkedIn post on {prompt}."
    prompt_template1 = ChatPromptTemplate.from_template(template1, inputVariable=["prompt"])

    # Second LLM Chain: Challenges of the LinkedIn Post
    template2 = """
    You have to give out the details of the product for LinkedIn post about {prompt}.
    Please follow the following guidelines:
        - use an eye-catching & interactive hook as an headline of the post in the content of the post,
        - Focus on the latest trends, key players, and significant news related to {prompt}.
        - Develop a detailed content outline that includes an introduction, key points, and a call to action.
        - Include SEO-friendly keywords, relevant hashtags, and data or resources to enhance the post's visibility and impact.
    """
    prompt_template2 = ChatPromptTemplate.from_template(template2, inputVariable=["prompt"])

    # Define the LLM Chains
    llm_chain1 = LLMChain(llm=llm1, prompt=prompt_template1)
    llm_chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

    # Sequential Chain expects one input to be passed through
    chain = SimpleSequentialChain(chains=[llm_chain1, llm_chain2])
    
    # Invoke the chain, passing the prompt
    response = chain.invoke({"input":prompt})

    return response
