from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

def pipeline_cv(splits, X, Y, pipeline):
    
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    
    reports = []
    for train, test in kfold.split(X, Y):
        fit = pipeline.fit(X.iloc[train], Y.iloc[train])
        prediction = fit.predict(X.iloc[test])
        
        reports.append(
            pd.DataFrame(
                metrics.classification_report(
                    Y.iloc[test],prediction,output_dict=True
                )
            )
        )

    df_concat = pd.concat([x for x in reports])

    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()

    return df_means

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def plot_confusion_matrix(true,predicted,classes):
    import itertools
    cm=confusion_matrix(true,predicted)
    
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    #plt.title('Confusion matrix',fontdict={'size':20})
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes,fontdict={'size':14})
    ax.set_yticklabels([''] + classes,fontdict={'size':14})
    plt.xlabel('Predicted',fontdict={'size':14})
    plt.ylabel('True',fontdict={'size':14})
    plt.grid(b=None)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             fontdict={'size':14,'weight':'heavy'},
             color="black" if cm[i, j] > thresh else "black")
    plt.show()
