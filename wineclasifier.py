
import warnings
warnings.filterwarnings('ignore')
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
# clasifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#from sklearn.metrics import r2_score, mean_squared_error, 
from sklearn.metrics import confusion_matrix, accuracy_score
import urllib 
import os

class WineMultiClassifier():
    def __init__(self, 
            download_url:str='https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', 
            download_file:str='data/winequality-white.csv',
            y_categories:list=[]
        ) -> None:
        self.download_url:str = download_url
        self.download_file:str = download_file
        self.y_categories:list = y_categories
        self.data:pd.DataFrame = None
        self.columns = ["Acidez fija", "Acidez volatil", 
            "Acido citrico", "Azucar residual", "Cloruros",
            "Dioxido de sulfuro libre", "Dioxido de sulfuro total",
            "Densidad", "pH","Sulfatos", "alcohol", "Calidad"]
        self.printfn = print
        # X and y raw data vectors
        self.X:np.array = None
        self.y:np.array = None
        # Tran and Test properties
        self.X_train:np.array = None 
        self.X_test:np.array = None 
        self.y_train:np.array = None 
        self.y_test:np.array = None 
        # Scaler
        self.scaler:StandardScaler = StandardScaler()

        # Dictionaty of models to train
        self.models:dict = dict()

    def do_all(self):
        """Donwload, read and split data. 
        Fit and transform X data.
        Load default models and train it.
        Fill metrics
        """
        self.data_download()
        self.data_read()
        self.train_test_split()
        self.feature_fit_and_tranform_X()
        self.add_default_models()
        self.train_models()
        self.fill_metrics()

    def add_model(self, key:str, model_def:dict) -> None:
        """Add a model to the model dict.
        The key os de key in the doct and the model_def is the definition expecting
        to have a dictionary with the structure
        {
            'name':'Model Name',
            'modelClass': model.class.pointer,
            'parameters':{
                'p1': 5,
                'p2': 'minkowski',
                'p3': 2
            }
        }
        """
        self.models[key] = model_def
        self.models[key]['model'] = model_def['modelClass'](**model_def['parameters'])
    
    def print(self, text:str):
        """Function to print and customie if needed"""
        self.printfn(text)

    def columns_as_md(self, header:str="Lista de variables") -> str:
        """Return the list of colums as  Mark Down string"""
        return "### {header}\n{text}".format(
            header=header, 
            text="\n".join(
                    ["    {} - {}".format(i+1, v) for i,v in enumerate(self.columns[:-1])]
                )
        )

    def models_as_md_short(self, header:str='List of Models') -> str:
        """Return the MarkDown of models in Short Format"""
        return "### {header}\n{text}".format(
            header=header, 
            text="\n".join(
                    ["- {}: {}".format(k, m['name']) for k,m in self.models.items()]
                )
        )

    def models_as_md_acc(self, header:str='Accuracy Score of Models') -> str:
        """Return the MarkDown of models in with Accuracy Score """
        return "### {header}\n{text}".format(
            header=header, 
            text="\n".join(
                    ["- {} => Accuracy:{:.2f} %".format(k, m['acc'] * 100) for k,m in self.models.items()]
                )
        )
    
    def models_as_md_kfold(self, header:str='K-Fold Cross Validation of Models') -> str:
        """Return the MarkDown of models in with K-Fold Cross Validation """
        return "### {header}\n{text}".format(
            header=header, 
            text="\n".join(
                    ["- {} => Accuracy: {:.2f} %  , std:{:2f} %".format(
                        k, 
                        m['kfold_stats']['mean'] * 100, 
                        m['kfold_stats']['std'] * 100
                        ) for k,m in self.models.items()]
                )
        )

    def data_download(self):
        """Function to download the file from the provided donwload_url"""
        if not os.path.isfile(self.download_file):
            urllib.request.urlretrieve(self.download_url, self.download_file)
            self.print("Successfully donwloaded file: {}".format(self.download_file))
        
    def data_read(self, sep:str=";") -> DataFrame:
        """read the file and dump into the data property as DataFrame"""
        if not os.path.isfile(self.download_file):
            self.print("File not found at: {}".format(self.download_file))
            return None
        self.data = pd.read_csv(self.download_file, sep=sep)
        return self.data

    def train_test_split(self, test_size:int=0.25, random_state:int=0) -> tuple:
        """Split the tran and test set, return a tuple with (X_train, X_test, y_train, y_test)"""
        self.X = self.data.iloc[:, :-1].values
        
        #check if is necesary to modify the categories in y
        if self.y_categories and len(self.y_categories) > 0:
            self.y_categories.sort(reverse=True)
            limt_sup = 0
            for i, limit in enumerate(self.y_categories):
                mask = self.data.iloc[:, -1] >= limit
                if i > 0:
                    mask = mask &  (self.data.iloc[:, -1] < limt_sup)
                self.data.iloc[mask, -1] = i
                limt_sup = limit
        self.y = self.data.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def feature_scale_fit(self):
        """Use the Scaler ton tranform the train and test set"""
        self.X_train = self.scaler.fit_transform(self.X_train)

    def feature_transform(self, X:np.array):
        """Transform data based on the fitted scaler, usually 
        calling the function 'feature_scale_fit'"""
        return self.scaler.transform(X)

    def feature_fit_and_tranform_X(self):
        """Feature Scale for and transform X_train and X_test"""
        self.feature_scale_fit()
        self.X_test = self.feature_transform(self.X_test)

    def add_default_models(self) -> None:
        """Add the default models for clasification
        - LogisticRegression
        - K-NN
        - SVM
        """
        default_models:dict = {
            'LogisticRegression':{
                'name':'Logistic Regression',
                'modelClass': LogisticRegression,
                'parameters':{
                    'random_state': 0
                }
            },
            'K-NN':{
                'name':'K Nearest Neighbors',
                'modelClass': KNeighborsClassifier,
                'parameters':{
                    'n_neighbors': 5,
                    'metric': 'minkowski',
                    'p': 2
                }
            },
            'SVM':{
                'name':'Support Vector Machine',
                'modelClass': SVC,
                'parameters':{
                    'kernel': 'linear',
                    'random_state': 0
                }
            },
            'Kernel SVM':{
                'name':'Kernel - Support Vector Machine',
                'modelClass': SVC,
                'parameters':{
                    'kernel': 'rbf',
                    'random_state': 0
                }
            },
            'Naive Bayes':{
                'name':'Naive Bayes',
                'modelClass': GaussianNB,
                'parameters':{}
            },
            'Decision Tree Classification':{
                'name':'Decision Tree Classification',
                'modelClass': DecisionTreeClassifier,
                'parameters':{
                    "criterion": 'entropy',
                    'random_state': 0
                }
            },
            'Random Forest Classification':{
                'name':'Random Forest Classification',
                'modelClass': RandomForestClassifier,
                'parameters':{
                    "criterion": 'entropy',
                    "n_estimators": 10,
                    'random_state': 0
                }
            }
        }
        for k,v in default_models.items():
            self.add_model(k,v)

    def train_model(self, key:str) -> None:
        """Train the desired model, foun by the key"""
        if key  in self.models.keys():
            self.models[key]['model'].fit(self.X_train, self.y_train)

    def train_models(self) -> None:
        """Train all the models added in the models dict"""
        for k in self.models.keys():
            self.train_model(k)

    def model_metrics(self, key:str) -> None:
        """Fill the confusion matrix and accuracy score of the selected model.
        The K-Fold class validation is calculated too"""
        if key  in self.models.keys():
            model = self.models[key]['model']
            y_pred = model.predict(self.X_test)
            self.models[key]['y_pred'] = y_pred
            self.models[key]['cm'] = confusion_matrix(self.y_test, y_pred)
            self.models[key]['acc'] = accuracy_score(self.y_test, y_pred)
            self.models[key]['kfold'] = cross_val_score(
                estimator = model, 
                X = self.X_train, 
                y = self.y_train, cv = 10)
            self.models[key]['kfold_stats'] = {
                "mean": self.models[key]['kfold'].mean(),
                "std": self.models[key]['kfold'].std()
            }


    def fill_metrics(self) -> None:
        """Fill the metric for consusion matrix and accuracy score"""
        for k in self.models.keys():
            self.model_metrics(k)

    def cm_as_plots(self, cols:int=2) -> tuple:
        """return  fig, axs"""
        last_blank_cols = len(self.models) % cols
        plot_rows = int(len(self.models) / cols) + last_blank_cols
        fig, axs = plt.subplots(plot_rows, cols, figsize=(16,16), tight_layout=True)
        fig.suptitle('Confussion Matrix for all Models', fontsize=18, fontweight='bold')
        fig.tight_layout()
        for i,k in enumerate(self.models.keys()):
            ax = axs.flat[i]
            confusion_matrix = self.models[k]['cm']
            ax = sns.heatmap(
                confusion_matrix ,
                annot=True, 
                annot_kws={"size": 12}, 
                ax=ax,
                cmap=plt.cm.Greens)
            ax.set_title("{}".format(k), fontsize=16, fontweight='bold')
            ax.set_xlabel("Predicciones", fontsize=12)
            ax.set_ylabel("Actuales", fontsize=12)
        for i in range(0, last_blank_cols):
            axs.flat[-(i+1)].axis('off') # clear existing plot
        return fig, axs







        
    
        