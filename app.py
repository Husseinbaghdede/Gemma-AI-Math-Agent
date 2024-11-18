import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI  

from dotenv import load_dotenv
import os


load_dotenv()
# Setup page config
st.set_page_config(page_title="Text to Math Problem Solver And Data Search Assistant")
st.title("Text to Math Problem Solver Using Google Gemma 2")

# API key handling
groq_api_key = st.sidebar.text_input(label="Groq Api Key", type="password")
if not groq_api_key:
    st.info("Please add your Groq Api Key to continue")
    st.stop()


# openai_api_key = os.getenv('OPENAI_API_KEY')

# llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model="Gemma2-9b-It")
# Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only Input Math expression needs to be provided"
)

# Initialize Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching information from Wikipedia"
)

# Create reasoning prompt
reasoning_prompt = """
You are agent tasked for solving users mathematical questions.
Logically arrive at the solution and provide detailed answer and display it point wise for the question below
Question : {question}
Answer:
"""

prompt_template = PromptTemplate(input_variables=['question'], template=reasoning_prompt)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)


st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)


assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
)  

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I am a Math chatbot who can answer all your math questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

question = st.text_area(
    "Enter your question",
    "I have 5 bananas and 7 grapes. I decided to buy 3 times as many bananas and twice as many grapes as I currently have. After buying these additional fruits, how many bananas and grapes do I have in total?"
)

# Process question
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            try:
                # Add question to chat history
                st.session_state.messages.append({"role": "user", "content": question})
                st.chat_message("user").write(question)
               
                response = assistant_agent({"input": question})["output"]
                
                # Add response to chat history
                st.session_state.messages.append({"role": 'assistant', 'content': response})
                
                st.write('### Response:')
                st.success(response)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try rephrasing your question or breaking it down into smaller parts.")
    else:
        st.warning("Please enter a question")
