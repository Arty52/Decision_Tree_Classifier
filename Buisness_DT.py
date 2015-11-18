"""
Authors: Art Grichine, Adam Beck
References: http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
"""

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

def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    code = []

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            code.append(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            code.append(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
            code.append(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")
                code.append(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)
    
    # Save code to file
    with open('code.txt', 'w') as file:
        for line in code:
            file.write('{}\n'.format(line))

def main():
    df = get_business_data()
    
    print("* df.head()", df.head(), sep="\n", end="\n\n")
    print("* df.tail()", df.tail(), sep="\n", end="\n\n")

    features = list(df.columns[1:8])
    print("* features:", features, sep="\n")

    y = df["Business_Category"]
    X = df[features]
    
    # Use min_samples_split=20 to declare leaf bin if sample < 20
    dt = DecisionTreeClassifier(min_samples_split=20, 
                                random_state=99, 
                                criterion='entropy')
    # dt = DecisionTreeClassifier(random_state=99, criterion='entropy')
    
    
    dt.fit(X,y)
    
    visualize_tree(dt, features)

    get_code(dt, features, ['Restaurant', 'Retail'])

if __name__ == "__main__":
    main()