# Python-Web-Scraping-Tool

Welcome to Pyhton Web Scraper! 🎉

This tool is designed to help you scrape research articles from Google Scholar and Pubmed using Python. It leverages Selenium and BeautifulSoup to extract valuable information, including article titles, authors, publication years, best performing models, dataset sizes, and more.

## Features

- 🌐 **Web Scraping**: Extracts data from Google Scholar and Pubmed, including article metadata and abstracts.
- 🔍 **Keyword Matching**: Identifies best performing machine learning models and algorithms within abstracts.
- 📊 **Data Extraction**: Captures dataset sizes and accuracy of top-performing models in the articles.
- 📁 **Yearly Data Organization**: Saves scraped data into an Excel file with separate sheets for each year.

## Installation

To set up the tool, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/scholar-web-scraper.git

2. **Navigate to the project directory**:

   ```bash
   cd scholar-web-scraper

3. **Install the required dependencies**:

   ```bash
   pip install selenium beautifulsoup4 pandas

4. **Download the appropriyour ate ChromeDriver**:

   ```bash
   Make sure your chromedriver matches your version of Chrome. You can download it from the ChromeDriver site. Place the chromedriver.exe file in your project directory.

## Usage

Once everything is set up, you can start scraping data. Here’s how:
1. Run the Google Scholar script:
      ```bash
      python scholar.py
2. Run the Pubmed script:
      ```bash
      python pubmed.py
3. Input your search query when prompted.
4. The script will automatically scrape Google Scholar or Pubmed for articles matching your query and save the results in an Excel file named scholar.xlsx or pubmed.xlsx.
5. The Excel file will contain separate sheets for each year from START to END years, with columns for titles, authors, publication years, best models mentioned, dataset sizes, and more.

## Contact
If you have any questions or feedback, feel free to reach out to me at suraj21209@iiitd.ac.in.
