import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from rapidfuzz import fuzz
import nltk

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load dataset
file_path = 'job_offers.csv'
df = pd.read_csv(file_path)

# Load stopwords
stop = stopwords.words('english')

# Streamlit App
st.title("MilkyWay Matchmakers")

st.header("Upload Resume:")
st.subheader("Option 1:")
text_1 = None

upload_file = st.file_uploader(label="Upload Your Resume in PDF format", type="pdf")
if upload_file is not None:
    from PyPDF2 import PdfReader
    reader = PdfReader(upload_file)
    text_1 = ""
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text_1 += page.extract_text()

st.subheader("Option 2:")
resume_text = st.text_area(label="Write Your Resume")

# Job Selection from Dataset
st.header("Select Job Description:")
job_options = df['job_title'].unique()
job_selected = st.selectbox('Choose a job role:', job_options)

# Extract job description from selected job
job_desc = df[df['job_title'] == job_selected]['job_Desc'].values[0]

# Preprocessing Functions
def remove_emojis(data):
    emoj = re.compile("[" 
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642" 
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def filter_text(text):
    text = text.lower()
    text = remove_emojis(text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuations
    return text

def preprocess_text(text):
    if text:  # Check if the text is not empty
        text = filter_text(text)
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in stop])
    return ""

# Preprocess job description and resume
filtered_job_text = preprocess_text(job_desc)
filtered_resume_text = preprocess_text(resume_text or text_1)

# TF-IDF Vectorization
def vectorize_texts(job_text, resume_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_text, resume_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Calculate similarity
similarity_score = vectorize_texts(filtered_job_text, filtered_resume_text)

# Show Results
st.subheader("Resume Score Based on Job Description (in %):")
score = similarity_score[0][0] * 100
st.write(f"Your resume matches the job by {score:.2f}%")

# Match Rating
if score < 20:
    st.write("It is a :red[BAD] match")
elif 20 <= score <= 40:
    st.write("It is an :orange[AVERAGE] match")
elif 40 < score <= 65:
    st.write("It is a :green[GOOD] match")
else:
    st.write("It is an :blue[EXCELLENT] match")
