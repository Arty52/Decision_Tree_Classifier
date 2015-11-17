import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def get_business_data():
    """Get the business data, from local csv."""
    if os.path.exists("Dataset.csv"):
        print("-- Dataset.csv found locally")
        df = pd.read_csv("Dataset.csv", index_col=0)

        with open("Dataset.csv", 'w') as f:
            print("-- writing to local Dataset.csv file")
            df.to_csv(f)

    return df

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def main():
    df = get_business_data()
    
    print("* df.head()", df.head(), sep="\n", end="\n\n")
    print("* df.tail()", df.tail(), sep="\n", end="\n\n")

    features = list(df.columns[1:8])
    print("* features:", features, sep="\n")

    y = df["Business_Category"]
    X = df[features]
    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99, criterion='entropy')
    # dt = DecisionTreeClassifier(random_state=99, criterion='entropy')
    dt.fit(X,y)
    
    visualize_tree(dt, features)
    

if __name__ == "__main__":
    main()