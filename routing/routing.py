### Router
import os, sys, re, json
from typing import Literal
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, load_prompt
from langchain_core.tools import StructuredTool, ToolException, Tool, tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.globals import set_debug
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from src.utils import read_keywords

# set_debug(True)
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(current_dir)
load_dotenv(os.path.join(kit_dir, '.env'))

# 1. load models
llm = SambaStudio(
        streaming=False,
        model_kwargs={
            'max_tokens_to_generate': 512,
            'select_expert': 'Mistral-7B-Instruct-v0.2', #'Meta-Llama-3-8B-Instruct',
            'temperature': 0.0,
            'repetition_penalty': 1.0,
            'top_k': 1,
            'top_p': 1.0,
            'do_sample': False,
        },
    )

# 2. create prompt
route_prompt = load_prompt(os.path.join(kit_dir,'routing/prompts/rag_routing_prompt_response_schema.yaml'))

# 3. create output parser
response_schemas = [
    ResponseSchema(name="datasource", description="choose vectorstore or llm"),
    ResponseSchema(
        name="explanation",
        description="explain the reason to choose this datasource.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 4. format prompt 
format_instructions = output_parser.get_format_instructions()
# load docs keywords 
keywords = read_keywords("routing/keywords/keywords_sambatune.pkl")
prompt = PromptTemplate(
    template=route_prompt.template,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions, "keywords": keywords},
)

# 5. create LCEL
question_router = prompt | llm | output_parser

# 6. user input
query = "who is the ceo of sambanova?"
# query = "What are the types of agent memory?"
results = question_router.invoke({'query': query})
print(results["datasource"])





