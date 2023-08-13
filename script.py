import sys
from PyPDF2 import PdfReader
import os, os.path
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import shutil
from pathlib import Path



print("\nName of Directory:", sys.argv[1])

def get_text_from_pdf(pdffile):
    # read text from pdf
    pdfreader = PdfReader(pdffile)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text




def get_keyword_bert(docs):
    try:
        vectorizer = KeyphraseCountVectorizer()
        kw_model = KeyBERT()
        onewords = kw_model.extract_keywords(docs=docs, vectorizer=vectorizer,top_n=5,stop_words='english')
        return onewords
    except:
        pass   

path = sys.argv[1]


def get_first_ele(x):
    res = []
    for i in x:
        res.append(i[0])
    return ','.join(res)



vectorizer = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidf_train.pkl", "rb")))

model_svm = pickle.load(open("resume-classifier-svm.pkl", "rb"))

file_locs = []
x_test_real = []
files = [f for f in os.listdir(path)]
for f in files:
    dir_path = path+f
    print(dir_path)
    if dir_path.split('.')[-1] == "pdf":
        file_locs.append(dir_path)
        text = get_text_from_pdf(dir_path)  
        keywords = get_first_ele(get_keyword_bert(text))
#         print(keywords)
        x_test_real.append(keywords)
    else:
        print(f"{dir_path} is not a pdf. We need file in pdf format. Thanks!!")

x_test_real_2 = vectorizer.fit_transform(x_test_real)
predictions = model_svm.predict(x_test_real_2)
print(predictions)


# Create directories
for f,p in zip(file_locs, predictions):
    print(f"{f}  -> {p}")
    source_pdf = rf"{f}"
    filename = f.split('/')[-1]
    destination_dir = rf"{os.getcwd()}/{p}"
    # Create the destination directory if it doesn't exist
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    # Move the PDF file to the destination directory
    shutil.move(source_pdf, os.path.join(destination_dir, os.path.basename(source_pdf)))
    print("PDF file moved successfully.")


df = pd.DataFrame(list(zip(file_locs, predictions)),
               columns =['filename', 'category'])
        
df.to_csv("categorized_resumes.csv")