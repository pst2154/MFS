# MFS 10k Analyzer

- Upload a 10k annual report
- Select report fields
- Get Report

## Change this code to start in the file in annual_report_analyzer.py
- use_BAM = True
- load_dotenv('/Users/alexsteiner/.env')
- API_KEY = os.getenv("BAM_KEY", None)
- API_ENDPOINT = os.getenv("BAM_API", None)

## Or If using GA
- use_BAM = False
- API_KEY = os.getenv("WXGA_API", None)
- API_ENDPOINT = os.getenv("WXGA_ENDPOINT", None)
- PROJECT_ID = os.getenv("WXGA_PROJECT", None)  

## To Start 
- streamlit run annual_report_analyzer.py