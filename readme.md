## Resume Categorization

## Instructions to run:
- First install the dependencies used in this project `pip install -r requirements.txt`

- Next run the script file using `python3 script.py $directory path$` The directory path should contain a batch of resumes in pdf format. It will not be processed if the resume is not in pdf format. I have assumed only pdf files to process the resumes.

## Delieverables

- script.py:  Main File for the project :D
- dataset.ipynb:   It collects data from html based resumes. I processed html content for data collection.

- dataset_prep_from_folders.ipynb:  It collects data from pdf files.

- Resume_classify.ipynb: After gathering and processing datas, I use the following steps to classify resumes: Keyword Extraction from resumes, tfidf vectorization, try different machine learning algorithms: Logistic Regression, Support Vector Machine, MLPClassifier, Random Forest Classifier, Multinomial Naive Bayes, Bert Classifier.


## Dataset

No of instances in dataframe = 4968 

No of classes = 24

Table:
| Category              | Numbers       |
| ----------------------| ------------- |
|INFORMATION-TECHNOLOGY |   240         |
|BUSINESS-DEVELOPMENT   |   240         |
|FINANCE                |   236         |           
|ADVOCATE               |   236         |
|ACCOUNTANT             |   236         |
|ENGINEERING            |   236         |
|CHEF                   |   236         |
|AVIATION               |   234         |
|FITNESS                |   234         |
|SALES                  |   232         |
|BANKING                |   230         |
|HEALTHCARE             |   230         |
|CONSULTANT             |   230         |
|CONSTRUCTION           |   224         |
|PUBLIC-RELATIONS       |   222         |
|HR                     |   220         |
|DESIGNER               |   214         |
|ARTS                   |   206         |
|TEACHER                |   204         |
|APPAREL                |   194         |
|DIGITAL-MEDIA          |   192         |
|AGRICULTURE            |   126         |
|AUTOMOBILE             |    72         |
|BPO                    |    44         |


I have split the dataset into training, validation and test set by using 70:10:20 ratio.


| Set             | Number of instances              |
| ----------------| ------------- |
|Training         |   3477        |
|Validation       |   497         |
|Test             |   994         | 


## Methodology

- Step 1: Keyword Extraction:  
I have extracted top 5 keywords from resume using KeyBert. However, It has taken significant time to extract keywords but the keywords I got using this technique is of high quality which might help the model to get better accuracy. One can use "rake" as fastest keyword extraction technique.

- Step 2: TF-IDF Vectorization
TF-IDF vectorization results in a matrix where each row represents a document, and each column represents a unique term in the corpus. The value in each cell of the matrix is the TF-IDF weight of the corresponding term in the respective document. 
I fit the extracted top 5 keywords to tf-idf vectorization

- Step 3: ML Models Exploration

Following table contains Model name and its accuracy on the validation set.

| Model                              | Accuracy      |
| -----------------------------------| ------------- |
|Logistic Regression               |   0.754         |
|Support Vector Machine           |   0.835         |
|Random Forest Classifier | 0.798|                                
|MLP               |   0.758         |
| Multinomial Naive Bayes|  0.76 |

I have taken SVM as best ML Model. I have tried Bert Classifier also but it is not giving accepted accuracy. 

I have tried different parameters for different ML models.
For SVM, I have used the following parameters:
'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['poly', 'rbf', 'sigmoid']

By using GridSearchCV, I have found best parameters for SVM classification. The parameters are 'C': 10, 'gamma': 1, 'kernel': 'rbf'.


