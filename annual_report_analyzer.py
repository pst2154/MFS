import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time
from pypdf import PdfReader
import pandas as pd
import math
import os
from dotenv import load_dotenv

from genai.model import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain import LangChainInterface

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods



import logging
logging.basicConfig(level=logging.INFO)
####################
# change this
#####################
# load_dotenv('/Users/alexsteiner/.env')
load_dotenv()
use_BAM = True

if use_BAM:
    API_KEY = os.getenv("BAM_KEY", None)
    API_ENDPOINT = os.getenv("BAM_API", None)
else:
    API_KEY = os.getenv("WXGA_API", None)
    API_ENDPOINT = os.getenv("WXGA_ENDPOINT", None)
    PROJECT_ID = os.getenv("WXGA_PROJECT", None)  

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if "process_doc" not in st.session_state:
    st.session_state.process_doc = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

############
# Vector DB
############
def get_vector_index(file, embeddings_use):
    loader = PyPDFLoader(file)
    documents = loader.load_and_split()

    # Split docs
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=5/100 * 500)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_use)

    # pdf_reader = PdfReader(file)
    # text = ""
    # for page in pdf_reader.pages:
    #     text += page.extract_text()

    # # split into chunks
    # text_splitter = CharacterTextSplitter(
    # separator="\n",
    # chunk_size=1024,
    # chunk_overlap=0.1 * 1024,
    # length_function=len
    # )

    # docs = text_splitter.split_text(text)
    # vectorstore = FAISS.from_texts(docs, embedding=embeddings_use)
    
    return vectorstore

############
# LLM Engine
############
def report_topic(topic, db):

    if use_BAM:    
        creds = Credentials(API_KEY, API_ENDPOINT)

        params = GenerateParams(
            decoding_method = "greedy",
            min_new_tokens = 1,
            max_new_tokens = 1000,
            #temperature = 0.7,
            #top_k = 50,
            #top_p = 1,
            repetition_penalty = 1.2
        )

        llm = LangChainInterface(
            model='meta-llama/llama-2-70b-chat', 
            credentials=creds,
            params=params
        )
    else:
        generate_params = {
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 1000,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.REPETITION_PENALTY: 1.2
        }

        model = Model(
            model_id="meta-llama/llama-2-70b-chat",
            credentials={
                "apikey": API_KEY,
                "url": API_ENDPOINT
            },
            params=generate_params,
            project_id=PROJECT_ID
        )

        llm = WatsonxLLM(model=model)


    relevant_docs = db.similarity_search(topic,3)
    relevant_content = ''
    for doc in reversed(relevant_docs):
        relevant_content += '```\n' + 'Content: ' + doc.page_content + '\n```\n'     

    prompt = '''<<SYS>>
You are a financial analyst who reads 10k files and writes reports on topics.
Base your answers on the Relevant Information Section only and do not make up any information.
Return answer using Bullet points like the following:

Report: 
•
•
•
<</SYS>>

[INST]
Relevant Information: 
{docs}

Topic: {topic}[/INST]
Report:'''

    

    to_send = prompt.format(docs = relevant_content.strip(), topic=topic)

    response = llm(to_send)

    return response


##########################################
# Attributes
##########################################
fiscal_year = {
    "performance_highlights": "Key performance highlights and financial statistics over the past year.",
    "major_events": "Highlight of significant events, acquisitions, or strategic shifts that occurred over the past year.",
    "challenges_encountered": "Challenges the company faced during the past year and, if and how they managed or overcame them."
}

fiscal_year_attributes = ["performance_highlights", "major_events", "challenges_encountered"]

strat_outlook = {
    "strategic_initiatives": "The company's primary objectives and growth strategies for the upcoming years.",
    "market_outlook": "Insights into the broader market, competitive landscape, and industry trends the company anticipates.",
    "product_roadmap": "Upcoming launches, expansions, or innovations the company plans to roll out."
}

strat_outlook_attributes = ["strategic_initiatives", "market_outlook", "product_roadmap"]

risk_management = {
    "risk_factors": "Primary future risks the company acknowledges.",
    "risk_mitigation": "Risk management strategies."
}

risk_management_attributes = ["risk_factors", "risk_mitigation"]

innovation = {
    "r_and_d_activities": "Overview of the company's focus on research and development, major achievements, or breakthroughs.",
    "innovation_focus": "Mention of new technologies, patents, or areas of research the company is diving into."
}

innovation_attributes = ["r_and_d_activities", "innovation_focus"]

    
##########################################
# report insights
##########################################
def report_insights(fields_to_include, section_num):

    fields = None
    attribs = None

    if section_num == 1:
        fields = fiscal_year
        attribs = fiscal_year_attributes
    elif section_num == 2:
        fields = strat_outlook
        attribs = strat_outlook_attributes
    elif section_num == 3:
        fields = risk_management
        attribs = risk_management_attributes
    elif section_num == 4:
        fields = innovation
        attribs = innovation_attributes

    ins = {}
    for i, field in enumerate(attribs):
        if fields_to_include[i]:
            qqq = field + ': ' + fields[field]
            logging.info('''**************************************
            ''' + qqq + '''
            ************************************************''')
            response = report_topic(topic=qqq, db = st.session_state.index)
            ins[field] = response

    return {
        "insights": ins
    }

####################
# The app 
#####################
st.set_page_config(page_title="Financial Health Summary", page_icon=":card_index_dividers:", initial_sidebar_state="expanded", layout="wide")

st.title("Financial Health Summary : Annual 10K Report")
st.info("""
1. Upload the annual report of your chosen company in PDF format. 
2. Click on 'Process PDF' to load the document.
3. Select the sections to report on.
4. Press the 'Analyze Report' button.
5. Wait a few minutes to get your report.
""")

for insight in fiscal_year_attributes:
    if insight not in st.session_state:
        st.session_state[insight] = None

for insight in strat_outlook_attributes:
    if insight not in st.session_state:
        st.session_state[insight] = None

for insight in risk_management_attributes:
    if insight not in st.session_state:
        st.session_state[insight] = None

for insight in innovation_attributes:
    if insight not in st.session_state:
        st.session_state[insight] = None

if "end_time" not in st.session_state:
    st.session_state.end_time = None

#############################
# Main Code
#############################
st.session_state.uploaded_file = st.sidebar.file_uploader("Upload the annual report in PDF format", type=["pdf"])

st.sidebar.info("""
Example reports you can upload here: 
- [Apple Inc.](https://s2.q4cdn.com/470004039/files/doc_financials/2022/q4/_10-K-2022-(As-Filed).pdf)
- [Microsoft Corporation](https://microsoft.gcs-web.com/static-files/07cf3c30-cfc3-4567-b20f-f4b0f0bd5087)
- [Tesla Inc.](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2022-Update)
""")

if st.session_state.uploaded_file is not None:
    logging.info('''**************************************
    Uploading document
    ************************************************''')
    st.session_state.file_uploaded = True

    file_name = st.session_state.uploaded_file.name
    save_path = os.path.join("data", file_name)
    st.session_state.save_path = save_path

    logging.info('Save Path: ' + st.session_state.save_path)

    if not os.path.exists(save_path):
        # Save the file only if it doesn't exist already
        with open(save_path, "wb") as f:
            f.write(st.session_state.uploaded_file.read())

if st.sidebar.button("Process Document"):
    if 'process_doc' not in st.session_state or not st.session_state.process_doc:

        if st.session_state.embeddings is None:
            logging.info('''**************************************
            Loaging HF Embeddings
            ************************************************''')
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1')


        st.session_state.process_doc = True
        logging.info('''**************************************
        Processing Document
        ************************************************''')
        with st.spinner("Processing Document..."):
            if st.session_state.uploaded_file is not None:
                # The code to process the document goes here
                print('file:' + str(st.session_state.uploaded_file.name))
                st.session_state.index = get_vector_index(st.session_state.save_path, st.session_state.embeddings)
                st.session_state.process_doc = True
                st.toast("Document Processed!")
            else:
                # Handle the case where no file is uploaded
                st.error("No file uploaded. Please upload a PDF document.")


if st.session_state.process_doc:

    col1, col2 = st.columns([0.25, 0.75])

    with col1:
        st.write("""
            ### Select Insights
        """)
        
        with st.expander("**Fiscal Year Highlights**", expanded=True):
            performance_highlights = st.toggle("Performance Highlights")
            major_events = st.toggle("Major Events")
            challenges_encountered = st.toggle("Challenges Encountered")

            fiscal_year_highlights_list = [performance_highlights, major_events, challenges_encountered]
        with st.expander("**Strategy Outlook and Future Direction**", expanded=True):
            strategic_initiatives = st.toggle("Strategic Initiatives")
            market_outlook = st.toggle("Market Outlook")
            product_roadmap = st.toggle("Product Roadmap")

            strategy_outlook_future_direction_list = [strategic_initiatives, market_outlook, product_roadmap]

        with st.expander("**Risk Management**", expanded=True):
            risk_factors = st.toggle("Risk Factors")
            risk_mitigation = st.toggle("Risk Mitigation")

            risk_management_list = [risk_factors, risk_mitigation]

        with st.expander("**Innovation and R&D**", expanded=True):
            r_and_d_activities = st.toggle("R&D Activities")
            innovation_focus = st.toggle("Innovation Focus")

            innovation_and_rd_list = [r_and_d_activities, innovation_focus]


    with col2:
        if st.button("Analyze Report"):
            logging.info('''**************************************
            Analyzing Report
            ************************************************''')
            start_time = time.time()

            with st.status("**Analyzing Report...**"):


                if any(fiscal_year_highlights_list):
                    st.write("Fiscal Year Highlights...")

                    for i, insight in enumerate(fiscal_year_attributes):
                        if st.session_state[insight]:
                            fiscal_year_highlights_list[i] = False

                    response = report_insights(fiscal_year_highlights_list, 1)

                    for key, value in response["insights"].items():
                        st.session_state[key] = value

                if any(strategy_outlook_future_direction_list):
                    st.write("Strategy Outlook and Future Direction...")

                    for i, insight in enumerate(strat_outlook_attributes):
                        if st.session_state[insight]:
                            strategy_outlook_future_direction_list[i] = False
                    response = report_insights(strategy_outlook_future_direction_list, 2)

                    for key, value in response["insights"].items():
                        st.session_state[key] = value


                if any(risk_management_list):
                    st.write("Risk Management...")

                    for i, insight in enumerate(risk_management_attributes):
                        if st.session_state[insight]:
                            risk_management_list[i] = False
                    
                    response = report_insights(risk_management_list, 3)

                    for key, value in response["insights"].items():
                        st.session_state[key] = value

                if any(innovation_and_rd_list):
                    st.write("Innovation and R&D...")

                    for i, insight in enumerate(innovation_attributes):
                        if st.session_state[insight]:
                            innovation_and_rd_list[i] = False

                    response = report_insights(innovation_and_rd_list, 4)
                    st.session_state.innovation_and_rd = response

                    for key, value in response["insights"].items():
                        st.session_state[key] = value

                st.session_state["end_time"] = time.time() - start_time



                st.toast("Report Analysis Complete!")
        
        if st.session_state.end_time:
            st.write("Report Analysis Time", st.session_state.end_time)


    # if st.session_state.all_report_outputs:
    #     st.toast("Report Analysis Complete!")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Fiscal Year Highlights", "Strategy Outlook and Future Direction", "Risk Management", "Innovation and R&D"])

        
            

        with tab1:
            st.write("## Fiscal Year Highlights")
            try: 
                if performance_highlights:
                    if st.session_state['performance_highlights']:
                        st.write("### Performance Highlights")
                        text = st.session_state['performance_highlights']
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("fiscal Year Highlights insight has not been generated")
            except:
                st.error("This insight has not been generated")

            try:
                if major_events:
                    if st.session_state["major_events"]:
                        st.write("### Major Events")
                        text = st.session_state["major_events"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                        
                    else:
                        st.error("Major Events insight has not been generated")
            except:
                st.error("This insight has not been generated")
            try:
                if challenges_encountered:
                    if st.session_state["challenges_encountered"]:
                        st.write("### Challenges Encountered")
                        text = st.session_state["challenges_encountered"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Challenges Encountered insight has not been generated")
            except:
                st.error("This insight has not been generated")
            # st.write("### Milestone Achievements")
            # st.write(str(st.session_state.fiscal_year_highlights.milestone_achievements))


        
        with tab2:
            st.write("## Strategy Outlook and Future Direction")
            try:
                if strategic_initiatives:
                    if st.session_state["strategic_initiatives"]:
                        st.write("### Strategic Initiatives")
                        text = st.session_state["strategic_initiatives"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Strategic Initiatives insight has not been generated")
            except:
                st.error("This insight has not been generated")

            try:
                if market_outlook:
                    if st.session_state["market_outlook"]:
                        st.write("### Market Outlook")
                        text = st.session_state["market_outlook"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Market Outlook insight has not been generated")

            except:
                st.error("This insight has not been generated")

            try:
                if product_roadmap:
                    if st.session_state["product_roadmap"]:
                        st.write("### Product Roadmap")
                        text = st.session_state["product_roadmap"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Product Roadmap insight has not been generated")
            except:
                st.error("This insight has not been generated")

        with tab3:
            st.write("## Risk Management")

            try:
                if risk_factors:
                    if st.session_state["risk_factors"]:
                        st.write("### Risk Factors")
                        text = st.session_state["risk_factors"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Risk Factors insight has not been generated")
            except:
                st.error("This insight has not been generated")

            try:
                if risk_mitigation:
                    if st.session_state["risk_mitigation"]:
                        st.write("### Risk Mitigation")
                        text = st.session_state["risk_mitigation"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Risk Mitigation insight has not been generated")
            except:
                st.error("This insight has not been generated")


        with tab4:
            st.write("## Innovation and R&D")

            try:
                if r_and_d_activities:
                    if st.session_state["r_and_d_activities"]:
                        st.write("### R&D Activities")
                        text = st.session_state["r_and_d_activities"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("R&D Activities insight has not been generated")
            except:
                st.error("This insight has not been generated")

            try:
                if innovation_focus:
                    if st.session_state["innovation_focus"]:
                        st.write("### Innovation Focus")
                        text = st.session_state["innovation_focus"]
                        st.markdown(f"```{text}", unsafe_allow_html=True)
                    else:
                        st.error("Innovation Focus insight has not been generated")
            except:
                st.error("This insight has not been generated")
