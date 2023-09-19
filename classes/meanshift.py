import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv('./datasets/candy.csv')
    print(dataset.head())
    
    X = dataset.drop('competitorname', axis=1)
    
    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)
    
    dataset['meansheft'] = meanshift.labels_
    print("="*64)
    print(dataset)