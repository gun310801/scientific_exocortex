# # from crewai.tools import tool
# import urllib.request
# from urllib.parse import quote
# import xml.etree.ElementTree as ET
# import json

# # @tool("publication_fetching_tool")
# def fetch_publications(argument: str) -> list:
#     """Fetches titles of publications from arXiv based on a search query."""
#     print("running publication tool")
#     print(f"[DEBUG] Tool input: {argument}")
#     query = quote(argument)
#     url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results=5"
    
#     try:
#         data = urllib.request.urlopen(url).read().decode("utf-8")
#     except Exception as e:
#         return f"[ERROR] Failed to fetch publications: {e}"

#     root = ET.fromstring(data)
#     ns = {"atom": "http://www.w3.org/2005/Atom"}

#     titles = []
#     for entry in root.findall("atom:entry", ns):
#         title = entry.find("atom:title", ns)
#         if title is not None:
#             titles.append(title.text.strip())

#     for link in entry.findall("atom:link", ns):
#         if link.attrib.get("title") == "pdf":
#             pdf_link = link.attrib["href"]

#     for title in titles:
#         summ = entry.find("atom:summary", ns)
#         if summ is not None:
#             open("publications.txt", "a").write(f"Title: {title} \nSummary: {summ.text.strip()}\n\n pdf_link: {pdf_link}")
#     if not titles:
#         return "No publications found."
#     return titles




from crewai.tools import tool
import urllib.request
from urllib.parse import quote
import xml.etree.ElementTree as ET
from pathlib import Path
from crewai_tools import RagTool

DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloaded_papers"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

@tool("publication_fetching_tool")
def fetch_publications(argument: str) -> list:
    """Fetches detailed publication data from arXiv and optionally downloads PDFs."""
    print("running publication tool")
    print(f"[DEBUG] Tool input: {argument}")

    download_pdfs = True 
    query = quote(argument)
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results=5"

    try:
        data = urllib.request.urlopen(url).read().decode("utf-8")
    except Exception as e:
        return f"[ERROR] Failed to fetch publications: {e}"

    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    publications = []
    downloaded_paths = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib["href"]

        pub_data = {
            "title": title.text.strip() if title is not None else "N/A",
            "pdf_link": pdf_link or "N/A"
        }

        if download_pdfs and pdf_link:
            try:
                safe_title = "".join(c for c in pub_data["title"] if c.isalnum() or c in (" ", "_", "-"))
                pdf_path = DOWNLOAD_DIR / f"{safe_title[:50]}.pdf"
                urllib.request.urlretrieve(pdf_link, str(pdf_path))
                pub_data["pdf_path"] = str(pdf_path)
                downloaded_paths.append(str(pdf_path))
                print(f"Downloaded: {pdf_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download PDF: {e}")

        publications.append(pub_data)

    
    if not publications:
        return "No publications found."
    if downloaded_paths:
        try:
            rag = RagTool(summarize=True,similarity_threshold=0.8, limit=5)
            for path in downloaded_paths:
                rag.add(data_type="file", path=path)
                print(f"Indexed: {path}")
            
            titles = [p['title'] for p in publications]
            return f"Successfully downloaded and indexed {len(downloaded_paths)} papers:\n" + "\n".join(f"- {title}" for title in titles)
        except Exception as e:
            print(f"[ERROR] Failed to index PDFs: {e}")
            return f"Downloaded {len(downloaded_paths)} papers but failed to index them: {e}"
    
    return f"Found {len(publications)} papers but could not download PDFs."

    

# if __name__ == "__main__":
#     print(fetch_publications('Classification ML models for nanoscience'))
