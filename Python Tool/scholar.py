from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pandas as pd
import re

model_keywords = [
    "Mol2vec", "DeePred-BBB", "LBM", "PNM", "FVM", "MLP", "Multilayer Perceptron", "CNN", 
    "Convolutional Neural Network", "RNN", "Recurrent Neural Network", "LSTM", 
    "Long Short-Term Memory", "GRU", "Gated Recurrent Unit", "BERT", "Bidirectional Encoder Representations from Transformers", 
    "GPT", "Generative Pre-trained Transformer", "SVM", "Support Vector Machine", "SVR", "Support Vector Regression", 
    "Random Forest", "XGBoost", "Linear Regression", "Decision Tree", "K-Nearest Neighbors", 
    "Naive Bayes", "Logistic Regression", "AdaBoost", "Gradient Boosting", "Perceptron", 
    "Autoencoder", "Principal Component Analysis", "Self-organizing Maps", "Restricted Boltzmann Machine", 
    "Hopfield Network", "Radial Basis Function Network", "K-Means", "Mean-Shift", 
    "DBSCAN", "Hierarchical Clustering", "Markov Models", "Hidden Markov Models", "Neuro-fuzzy Systems", 
    "Extreme Learning Machine", "Echo State Network", "Long Short-Term Memory Network", 
    "Recurrent Convolutional Neural Network", "Deep Belief Network", "Generative Adversarial Network", 
    "Variational Autoencoder", "Capsule Network", "Attention Mechanism", "Transformer", 
    "Residual Network", "Inception Network", "MobileNet", "DenseNet", "AlexNet", 
    "VGG", "LeNet", "GoogLeNet", "SqueezeNet", "DART", "U-Net", "FCN", 
    "Mask R-CNN", "YOLO", "Bert", "GPT-2", "GPT-3", "VQ-VAE", 
    "WaveNet", "Deep Q-Network", "Policy Gradient Methods", "Reinforcement Learning", 
    "Q-Learning", "SARSA", "A3C", "Dueling DQN", "Proximal Policy Optimization", 
    "AlphaZero", "Word2Vec", "Doc2Vec", "FastText", "Tfidf", 
    "Word Embeddings", "Neural Style Transfer", "CycleGAN", "Pix2Pix", "Wasserstein GAN", 
    "VAE-GAN", "StarGAN", "StyleGAN", "BigGAN", "BERT Variants", 
    "XLNet", "RoBERTa", "T5", "ERNIE", "MT-DNN", "DeBERTa", 
    "CTRL", "ALBERT", "ELECTRA", "GPT-4", "GPT-5", "SVM Variants", 
    "Random Forest Variants", "XGBoost Variants", "LightGBM", "CatBoost", 
    "AutoML", "H2O.ai", "TPOT", "Auto-sklearn", "PyCaret", "DeepCaret", 
    "Ludwig", "Transparency AI", "Featuretools", "Yellowbrick", 
    "SHAP", "LIME", "ELI5", "Scikit-learn", "Keras", 
    "TensorFlow", "PyTorch", "Caffe", "MXNet", "Theano", "CNTK", 
    "Chainer", "PaddlePaddle", "TFLite", "ONNX", "Apache Spark MLlib", 
    "H2O.ai", "DataRobot", "RapidMiner", "KNIME", "Orange", "Alteryx", 
    "Databricks", "Domino Data Lab", "IBM Watson Studio", "Google AI Platform", 
    "Microsoft Azure Machine Learning", "Amazon SageMaker", "Auto-Keras", 
    "AutoML-Zero", "EfficientNet", "MobileNetV2", "NASNet", "ProxylessNAS", 
    "MnasNet", "EfficientDet", "ResNeSt", "T-NAS", "GhostNet", 
    "NFNet", "RegNet", "Vision Transformers (ViT)", "BYOL", "SimCLR", 
    "SwAV", "CLIP", "DALL-E", "OpenAI GPT-3", "Turing-NLG", 
    "MADDPG", "SAC", "TRPO", "PPO", "DDPG", "A2C", 
    "A3C", "TD3", "D4PG", "Soft Actor-Critic", "Hindsight Experience Replay", 
    "H-DQN", "Distributed Distributional Deterministic Policy Gradients (D4PG)", 
    "Multi-Agent Actor-Critic (MAAC)", "Advantage Actor-Critic (A2C)", 
    "Q-Prop", "DDPG Variants", "Policy Gradient Variants", 
    "AlphaGo", "AlphaZero", "MuZero", "Chess AI (Stockfish, AlphaBeta)", 
    "Atari AI", "OpenAI Gym", "CARLA", "Unreal Engine", 
    "Unity ML-Agents", "ROS", "PyBullet", "Deep Reinforcement Learning", 
    "TabNet", "RuleFit", "CatBoost", "LightGBM", "XGBoost", 
    "H2O.ai", "Random Forest", "Gradient Boosting", "PyCaret", 
    "Auto-sklearn", "Neptune.ai", "Comet.ml", "DVC", "Mlflow", 
    "DataRobot", "Kubeflow", "Metaflow", "Cortex", "Kubeflow Pipelines", 
    "AI Fairness 360", "Google Fairness Indicators", "Aequitas", 
    "IBM Watson OpenScale", "Microsoft Fairlearn", "Explainable AI", 
    "SHAP", "LIME", "Extra-tree classifiers", "deep neural network", 
    "LogBB_Pred", "Deep-B3"
]

# path to ChromeDriver executable
chromedriver_path = '/chromedriver'

# Create ChromeOptions object and set the executable path
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument(f"executable_path={chromedriver_path}")

# Initialize the Chrome driver with ChromeOptions
driver = webdriver.Chrome(options=chrome_options)

# Define the search query for Google Scholar
search_query = input()

# Create a dictionary to store data for each year
data_by_year = {}

def find_data_size(text):
    # Split the text into lines
    lines = text.split('. ')
    
    max_value = None
    line_with_max_value = None

    for line in lines:
        # Use regular expressions to find all numeric values in the line
        numeric_values = re.findall(r'\d+(?:\.\d+)?', line)

        if numeric_values:
            # Convert the numeric values to float and find the maximum
            line_max_value = max(float(value) for value in numeric_values)

            if max_value is None or line_max_value > max_value:
                max_value = line_max_value
                line_with_max_value = line

    return line_with_max_value

def find_top_performing_model(abstract_text):
    # Split the text into lines
    lines = abstract_text.split('. ')
    line_with_top_model = ""

    for line in lines:
        if re.search(r'\b(?:top|best|leading|outperformed|performance|highest)\b', line, flags=re.IGNORECASE):
            line_with_top_model += line

    return line_with_top_model

# Loop through each year from 2010 to 2023
for year in range(2010, 2024):
    year_data = []

    # Inside the loop for each page
    for page in range(1, 6):
        # Open Google Scholar
        driver.get(f"https://scholar.google.com/scholar?start={(page-1)*10}&q={search_query}&as_ylo={year}&as_yhi={year}")

        # Wait for the results to load
        time.sleep(5)

        # Scroll down to load more results (you can adjust the number of scrolls)
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Get the page source and parse it with BeautifulSoup
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract information from the search results
        results = soup.find_all("div", {"class": "gs_ri"})
        
        # Loop through the results and extract information
        for result in results:
            title = result.find("h3", {"class": "gs_rt"}).text
            authors = result.find("div", {"class": "gs_a"}).text
            link = result.find("h3", {"class": "gs_rt"}).a['href']
            snippet = result.find("div", {"class": "gs_rs"}).text

            # Open the link to the article
            driver.get(link)
            time.sleep(5)  # Wait for the article page to load

            # Get the article page source and parse it with BeautifulSoup
            article_page_source = driver.page_source
            article_soup = BeautifulSoup(article_page_source, 'html.parser')

            # Get the abstract (snippet in this case)
            abstract = snippet

            data_set = find_data_size(abstract)
            top_model = find_top_performing_model(abstract)

            # Extract the entire text content of the article page
            article_text = article_soup.get_text()
            article_upper = article_text.upper()

            abstract_upper = abstract.upper()
            models_mentioned = []

            # Iterate through the model keywords and check for mentions in the article text
            for model in model_keywords:
                model_upper = model.upper()
                if model_upper in abstract_upper or f"({model_upper})" in abstract_upper:
                    models_mentioned.append(model)

            # If models are mentioned, join them into a comma-separated string
            if models_mentioned:
                model = ", ".join(models_mentioned)
            else:
                model = "No Model Found"
            if len(top_model) == 0:
                top_model = model

            # Append the information to the year_data list as a dictionary
            year_data.append({"Title": title, "Year": year, "Authors": authors, "Link": link, "Model": model, "Abstract": abstract, "Dataset Size": data_set, "Top Performing Model": top_model})

    # Store the year's data in the data_by_year dictionary
    data_by_year[year] = year_data

# Close the browser
driver.quit()

# Create an Excel writer object
excel_writer = pd.ExcelWriter("scholar.xlsx")

# Write each year's data to a separate sheet in the Excel file
for year, year_data in data_by_year.items():
    df = pd.DataFrame(year_data)
    df.to_excel(excel_writer, sheet_name=str(year), index=False)

# Save the Excel file
excel_writer._save()

print(f"Scraped data saved to scholar.xlsx with separate sheets for each year.")
