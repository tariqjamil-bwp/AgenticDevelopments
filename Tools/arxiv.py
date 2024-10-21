import requests
import os
from tqdm import tqdm
import arxiv
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

class Papers:
    def __init__(self, config):
        self.config = config

    def create_data_folder(self, download_path):
        if not os.path.exists(download_path):
            os.makedirs(download_path)
            print("Output folder created")
        else:
            print("Output folder already exists.")

    def download_papers(self, search_query, download_path, server="arxiv", start_date=None, end_date=None):
        self.create_data_folder(download_path)
        
        if server == "arxiv":
            client = arxiv.Client()

            search = arxiv.Search(
                query=search_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            results = list(client.results(search))
            for paper in tqdm(results):
                if os.path.exists(download_path):
                    paper_title = (paper.title).replace(" ", "_")
                    paper.download_pdf(dirpath=download_path, filename=f"{paper_title}.pdf")
                    print(f"{paper.title} Downloaded.")

        elif server == "medrxiv":
            if not start_date or not end_date:
                print("Error: 'start_date' and 'end_date' are required for medRxiv.")
                return

            # Construct the API URL
            api_url = f"https://api.medrxiv.org/details/{server}/{start_date}/{end_date}/0/json"

            response = requests.get(api_url)
            if response.status_code != 200:
                print(f"Failed to retrieve data from MedRxiv API. Status code: {response.status_code}")
                return

            data = response.json()

            if 'collection' not in data or len(data['collection']) == 0:
                print("No papers found with the given search query.")
                return

            papers = data['collection']

            for paper in tqdm(papers):
                title = paper['title'].strip().replace(" ", "_").replace("/", "_")  # Replace spaces and slashes with underscores
                pdf_url = f"https://www.medrxiv.org/content/{paper['doi']}.full.pdf"
                print(f"Attempting to download {title} from {pdf_url}")

                try:
                    pdf_response = requests.get(pdf_url)
                    if pdf_response.status_code == 200:
                        pdf_path = os.path.join(download_path, f"{title}.pdf")
                        with open(pdf_path, 'wb') as pdf_file:
                            pdf_file.write(pdf_response.content)
                        print(f"{title} Downloaded to {pdf_path}.")
                    else:
                        print(f"Failed to download {title}. Status code: {pdf_response.status_code}")
                except Exception as e:
                    print(f"An error occurred while downloading {title}: {e}")

        else:
            print(f"Server '{server}' is not supported.")
    

# set the parameters
query = "heart failure exercise tolerance"  # The topic you are interested in
server = "medrxiv"  # Set server to "medrxiv" or "arxiv"
start_date = "2003-07-31"  # Required for medRxiv, it is useful becacuse you don't want outdated papers
end_date = "2024-08-01"  # Required for medRxiv, it is useful becacuse you don't want outdated papers
# file path
config_file = "config.yml"

# Load the configuration file 
with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)

data = Papers(config)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def download_papers_with_retry(data, search_query, download_path, server="arxiv", start_date=None, end_date=None):
    data.download_papers(search_query=search_query, download_path=download_path, server=server, start_date=start_date, end_date=end_date)

# download papers
if query:
    download_papers_with_retry(data, query, config["data_path"], server=server, start_date=start_date, end_date=end_date)
