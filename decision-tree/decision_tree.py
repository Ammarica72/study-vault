import numpy as np  # numerical operations (log2, sums)
import pandas as pd  # dataframe handling
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D




def get_midpoints(df, column):
    """
    Compute threshold candidates by calculating midpoints between
    consecutive unique values in a dataframe column.

    This function is typically used when searching for possible split
    points in decision tree algorithms. The returned midpoints can be
    used as candidate thresholds for dividing continuous features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the feature column.

    column : str
        Name of the column for which midpoint thresholds are calculated.
        The column should contain numeric values.

    Returns
    -------
    list
        A list of midpoint values between consecutive unique sorted
        values in the specified column.

    Notes
    -----
    - Missing values are ignored.
    - Duplicate values are removed before calculating midpoints.
    - If the column contains fewer than two unique values, an empty list
      is returned because no midpoint can be computed.
    """

    # Sort unique non-missing values
    values = sorted(df[column].dropna().unique())
    
    # Compute midpoints between consecutive values
    midpoints = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
    
    return midpoints




def entropy(y):
    """
    Compute the Shannon entropy of a target variable.

    Entropy is a measure of uncertainty or impurity in the class
    distribution of a node. It is commonly used in decision tree
    algorithms to evaluate the quality of candidate splits.

    Parameters
    ----------
    y : pandas.Series
        A one-dimensional array or Series containing the class labels
        for the samples in a node.

    Returns
    -------
    float
        The Shannon entropy of the class distribution. An entropy of
        zero indicates a pure node, while larger values indicate
        greater class uncertainty.

    Notes
    -----
    The entropy is computed using:

        H = -Σ p_i log₂(p_i)

    where p_i is the probability of class i.
    """

    # Count the number of occurrences of each class
    counts = y.value_counts()

    # Convert class counts into class probabilities
    probs = counts / len(y)

    # Compute the Shannon entropy
    return -np.sum(probs * np.log2(probs))




def entropy_split(df, feature, threshold, target="LoanDecision"):
    """
    Compute the weighted entropy of a candidate decision tree split.

    This function evaluates a potential split by partitioning the dataset
    into two child nodes using the specified feature and threshold. The
    entropy of each child node is computed and combined as a weighted
    average based on the number of samples in each partition.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the feature and target columns.

    feature : str
        Name of the feature used to split the data.

    threshold : float or int
        Threshold value used to partition the feature into left
        (<= threshold) and right (> threshold) child nodes.

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    float
        The weighted entropy of the proposed split. Lower values
        indicate a better separation of the classes.

    Notes
    -----
    The weighted entropy is computed as:

        H_split = (N_left / N) * H_left
                + (N_right / N) * H_right

    where:
        - N is the total number of samples.
        - N_left and N_right are the numbers of samples in the left
          and right child nodes, respectively.
        - H_left and H_right are the Shannon entropies of the child
          nodes.
    """

    # Split the target values according to the threshold
    left = df[df[feature] <= threshold][target]

    right = df[df[feature] > threshold][target]

    # Total number of samples
    n = len(df)

    # Number of samples in each child node
    n_left = len(left)

    n_right = len(right)

    # Compute the entropy of each child node
    h_left = entropy(left) if n_left > 0 else 0

    h_right = entropy(right) if n_right > 0 else 0

    # Compute the weighted average entropy of the split
    weighted_entropy = (n_left / n) * h_left + (n_right / n) * h_right

    return weighted_entropy





def gini(y):
    """
    Compute the Gini impurity of a target variable.

    Gini impurity measures the probability of incorrectly classifying
    a randomly selected sample if its class label is assigned according
    to the class distribution within the node. It is commonly used in
    decision tree algorithms to evaluate the quality of candidate splits.

    Parameters
    ----------
    y : pandas.Series
        A one-dimensional array or Series containing the class labels
        for the samples in a node.

    Returns
    -------
    float
        The Gini impurity of the class distribution. A value of zero
        indicates a pure node containing only one class, while larger
        values indicate greater class impurity.

    Notes
    -----
    The Gini impurity is computed using:

        G = 1 - Σ p_i²

    where p_i is the probability of class i. The squared probability
    p_i² represents the probability that two independent selections
    belong to the same class i. Therefore, the Gini impurity is the
    complementary probability that the two selections belong to
    different classes.
    """

    # Count the number of occurrences of each class
    counts = y.value_counts()

    # Convert class counts into class probabilities
    probs = counts / len(y)

    # Compute the Gini impurity
    return 1 - np.sum(probs ** 2)


def gini_split(df, feature, threshold, target="LoanDecision"):
    """
    Compute the weighted Gini impurity of a candidate decision tree split.

    This function evaluates a potential split by partitioning the dataset
    into two child nodes using the specified feature and threshold. The
    Gini impurity of each child node is computed and combined as a
    weighted average based on the number of samples in each partition.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the feature and target columns.

    feature : str
        Name of the feature used to split the data.

    threshold : float or int
        Threshold value used to partition the feature into left
        (<= threshold) and right (> threshold) child nodes.

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    float
        The weighted Gini impurity of the proposed split. Lower values
        indicate a better separation of the classes.

    Notes
    -----
    The weighted Gini impurity is computed as:

        G_split = (N_left / N) * G_left
                + (N_right / N) * G_right

    where:
        - N is the total number of samples.
        - N_left and N_right are the numbers of samples in the left
          and right child nodes, respectively.
        - G_left and G_right are the Gini impurities of the child
          nodes.

    The split that minimizes the weighted Gini impurity is considered
    the best candidate because it produces the purest child nodes.
    """

    # Split the target values according to the threshold
    left = df[df[feature] <= threshold][target]

    right = df[df[feature] > threshold][target]

    # Total number of samples
    n = len(df)

    # Number of samples in each child node
    n_left = len(left)

    n_right = len(right)

    # Compute the Gini impurity of each child node
    g_left = gini(left) if n_left > 0 else 0

    g_right = gini(right) if n_right > 0 else 0

    # Compute the weighted average Gini impurity of the split
    weighted_gini = (n_left / n) * g_left + (n_right / n) * g_right

    return weighted_gini




def best_split_table(df, features, target="LoanDecision"):
    """
    Determine the optimal split for each feature using both entropy and
    Gini impurity.

    This function evaluates every candidate split threshold (computed as
    the midpoint between consecutive feature values) for each feature in
    the dataset. For each feature, it identifies the threshold that
    minimizes the weighted entropy and the threshold that minimizes the
    weighted Gini impurity.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the predictor features and the
        target variable.

    features : list of str
        List of feature names to evaluate as candidate splitting
        variables.

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    pandas.DataFrame
        A summary table containing one row per feature with the
        following information:

        - Feature : Name of the evaluated feature.
        - BestEntropyThreshold : Threshold producing the minimum
          weighted entropy.
        - MinEntropy : Minimum weighted entropy obtained.
        - BestGiniThreshold : Threshold producing the minimum weighted
          Gini impurity.
        - MinGini : Minimum weighted Gini impurity obtained.

    Notes
    -----
    For each feature, the algorithm:

    1. Computes all candidate thresholds using the midpoints between
       consecutive unique values.
    2. Evaluates the weighted entropy for every threshold.
    3. Evaluates the weighted Gini impurity for every threshold.
    4. Records the threshold that minimizes each impurity measure.

    The resulting table provides a direct comparison of the best split
    for each feature according to both entropy and Gini impurity.
    """

    results = []

    for feature in features:

        # Generate candidate split thresholds
        midpoints = get_midpoints(df, feature)

        # Initialize best impurity values
        best_ent = float("inf")
        best_gini = float("inf")

        # Initialize corresponding thresholds
        best_ent_t = None
        best_gini_t = None

        # Evaluate each candidate threshold
        for t in midpoints:

            h = entropy_split(df, feature, t, target)
            g = gini_split(df, feature, t, target)

            # Update best entropy split
            if h < best_ent:
                best_ent = h
                best_ent_t = t

            # Update best Gini split
            if g < best_gini:
                best_gini = g
                best_gini_t = t

        # Store the best results for the current feature
        results.append({
            "Feature": feature,
            "BestEntropyThreshold": best_ent_t,
            "MinEntropy": best_ent,
            "BestGiniThreshold": best_gini_t,
            "MinGini": best_gini
        })

    # Return the summary table
    return pd.DataFrame(results)



def plot_split(df, feature, threshold, target="LoanDecision"):
    """
    Visualize the class distribution resulting from a candidate decision
    tree split.

    This function partitions the dataset into left and right child nodes
    using the specified feature and threshold, then displays the class
    distribution of each child node as a bar chart. The visualization
    provides an intuitive assessment of how effectively the split
    separates the target classes.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the feature and target columns.

    feature : str
        Name of the feature used to split the data.

    threshold : float or int
        Threshold value used to partition the feature into left
        (<= threshold) and right (> threshold) child nodes.

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    None
        Displays two bar charts representing the class distributions of
        the left and right child nodes.

    Notes
    -----
    The function performs the following steps:

    1. Splits the dataset into left and right child nodes.
    2. Counts the number of samples belonging to each class in both
       nodes.
    3. Displays the class distributions side-by-side using bar charts.

    This visualization is useful for interpreting the quality of a
    candidate split before constructing a decision tree.
    """

    # Partition the target values into left and right child nodes
    left = df[df[feature] <= threshold][target]
    right = df[df[feature] > threshold][target]

    # Count the occurrences of each class in both child nodes
    left_counts = left.value_counts()
    right_counts = right.value_counts()

    # Create side-by-side bar charts
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Left child node
    ax[0].bar(left_counts.index, left_counts.values)
    ax[0].set_title(f"Below threshold of {threshold}")

    # Right child node
    ax[1].bar(right_counts.index, right_counts.values)
    ax[1].set_title(f"Above threshold of {threshold}")

    # Display the figure
    plt.tight_layout()
    plt.ion()
    plt.show()
    




def plot_branch_2d(df, x_feature, y_feature,
                   x_threshold, y_threshold,
                   target="LoanDecision"):
    """
    Visualize a two-level decision tree branch using a two-dimensional
    scatter plot.

    This function displays the dataset using two selected features as
    the x- and y-axes. Candidate decision boundaries are represented by
    a vertical line (split on the x-axis feature) and a horizontal line
    (split on the y-axis feature), illustrating how a decision tree
    partitions the feature space.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the predictor features and target
        variable.

    x_feature : str
        Feature plotted on the horizontal axis and used for the vertical
        decision boundary.

    y_feature : str
        Feature plotted on the vertical axis and used for the horizontal
        decision boundary.

    x_threshold : float or int
        Threshold value for the split on the x-axis feature.

    y_threshold : float or int
        Threshold value for the split on the y-axis feature.

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    None
        Displays a two-dimensional scatter plot with the corresponding
        decision boundaries.

    Notes
    -----
    The visualization performs the following steps:

    1. Plots each sample using the selected x- and y-axis features.
    2. Colors the samples according to their class labels.
    3. Draws a vertical decision boundary corresponding to the
       x-feature threshold.
    4. Draws a horizontal decision boundary corresponding to the
       y-feature threshold.

    The resulting figure provides an intuitive visualization of how
    sequential binary splits partition the feature space in a decision
    tree.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Map class labels to colors
    colors = df[target].map({"Approve": "green", "Reject": "red"})

    # Plot all samples
    ax.scatter(
        df[x_feature],
        df[y_feature],
        c=colors,
        alpha=0.7
    )

    # Draw the split on the x-axis feature
    ax.axvline(
        x=x_threshold,
        color="blue",
        linestyle="--",
        label=f"{x_feature} split"
    )

    # Draw the split on the y-axis feature
    ax.axhline(
        y=y_threshold,
        color="purple",
        linestyle="--",
        label=f"{y_feature} split"
    )

    # Label the axes and title
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title("Branch Split Visualization")

    # Display the legend and figure
    ax.legend()
    plt.ion()
    plt.show()




def find_pure_thresholds(df, features, target="LoanDecision"):
    """
    Identify split thresholds that produce a pure child node.

    This function examines every candidate threshold for each specified
    feature and determines whether the resulting left or right child
    node contains samples from only a single class. For each feature,
    the function retains the threshold that produces the largest pure
    subset.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the predictor features and target
        variable.

    features : list of str
        List of feature names to evaluate for candidate split
        thresholds.

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    pandas.DataFrame
        A summary table containing one row for each feature that
        produces a pure subset. The table includes:

        - Feature : Feature used for the split.
        - Threshold : Threshold producing the pure subset.
        - Subset : Indicates whether the pure subset is below
          ("Below") or above ("Above") the threshold.
        - Class : Class label of the pure subset.
        - Samples : Number of samples contained in the pure subset.

    Notes
    -----
    For each feature, the algorithm:

    1. Computes all candidate thresholds using the midpoints between
       consecutive unique feature values.
    2. Evaluates the subset below each threshold.
    3. Evaluates the subset above each threshold.
    4. Determines whether either subset is pure (contains only one
       unique class).
    5. Retains the threshold producing the largest pure subset.

    This function is useful for identifying highly discriminative
    decision boundaries that completely separate one class from the
    remaining data.
    """

    results = []

    # Evaluate each feature independently
    for feature in features:

        best = None
        max_samples = -1

        # Evaluate every candidate threshold
        for t in get_midpoints(df, feature):

            # Evaluate the lower (left) subset
            lower = df[df[feature] <= t]

            if len(lower) > 0 and lower[target].nunique() == 1:
                if len(lower) > max_samples:
                    max_samples = len(lower)
                    best = {
                        "Feature": feature,
                        "Threshold": t,
                        "Subset": "Below",
                        "Class": lower[target].iloc[0],
                        "Samples": len(lower)
                    }

            # Evaluate the upper (right) subset
            upper = df[df[feature] > t]

            if len(upper) > 0 and upper[target].nunique() == 1:
                if len(upper) > max_samples:
                    max_samples = len(upper)
                    best = {
                        "Feature": feature,
                        "Threshold": t,
                        "Subset": "Above",
                        "Class": upper[target].iloc[0],
                        "Samples": len(upper)
                    }

        # Store the best pure subset for the current feature
        if best is not None:
            results.append(best)

    # Return the summary table
    return pd.DataFrame(results)



def apply_pure_rules(df, rules, target="LoanDecision"):
    """
    Apply pure classification rules to a dataset and remove the samples
    correctly classified by those rules.

    This function sequentially applies a set of pure decision rules,
    where each rule identifies a subset of samples belonging entirely
    to a single class. Samples satisfying both the rule condition and
    the associated class label are removed from the dataset. The
    remaining samples can then be used for subsequent iterations of
    decision tree construction.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing the predictor features and target
        variable.

    rules : pandas.DataFrame
        A table of pure classification rules. Each rule must contain
        the following columns:

        - Feature : Feature used for the split.
        - Threshold : Threshold defining the split.
        - Subset : Indicates whether the rule applies to values
          "Below" (<= threshold) or "Above" (> threshold).
        - Class : Class label assigned by the rule.
        - Samples : Number of samples represented by the rule
          (informational only).

    target : str, optional
        Name of the target (class label) column.
        Default is "LoanDecision".

    Returns
    -------
    pandas.DataFrame
        A new dataframe containing only the samples that were not
        classified by any of the supplied pure rules.

    Notes
    -----
    The function processes the rules sequentially. For each rule:

    1. Identifies the subset of samples satisfying the threshold
       condition.
    2. Determines which of those samples belong to the specified
       class.
    3. Removes those correctly classified samples from the dataset.

    The remaining dataset represents the observations that still
    require further decision tree splitting.
    """

    # Create a working copy of the dataset
    remaining = df.copy()

    # Apply each pure classification rule
    for _, rule in rules.iterrows():

        feature = rule["Feature"]
        threshold = rule["Threshold"]
        subset = rule["Subset"]
        class_value = rule["Class"]

        # Determine the subset specified by the rule
        if subset == "Below":
            mask = (remaining[feature] <= threshold)
        else:  # Above
            mask = (remaining[feature] > threshold)

        # Remove samples correctly classified by the rule
        remaining = remaining[~(mask & (remaining[target] == class_value))]

    # Return the remaining unclassified samples
    return remaining

def latex_df(df, cap, label):
    """
    Converts a pandas DataFrame into a LaTeX table format.

    This function uses the DataFrame's built-in `to_latex()` method to
    generate a LaTeX representation of the table, prints the generated
    LaTeX code, and returns it.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame that will be converted into a LaTeX table.

    cap : str
        The caption text for the LaTeX table.

    label : str
        The LaTeX label used for referencing the table (e.g., in a report
        or document).

    Returns
    -------
    str
        A string containing the LaTeX code representation of the DataFrame.

    Notes
    -----
    The DataFrame index is excluded from the generated LaTeX table.
    """

    # This converts the df (table) to latex. 
    latex_table = df.to_latex(
        index=False,
        caption=cap,
        label=label
    )
    print(latex_table)
    
    return latex_table

def check_leaf_node(df, df_name, label_col):
    """
    Checks whether a decision tree node is a leaf node based on class purity.

    A node is considered a leaf if all samples in the node belong to the
    same class. This is detected by comparing the largest class count with
    the total number of samples in the node.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset corresponding to the current decision tree node.

    df_name : str
        A name identifier for the dataframe/node used for reporting.

    label_col : str
        The name of the column containing the target class labels.

    Returns
    -------
    None
        Prints whether the node is a leaf node. If it is not a leaf, it
        prints the total number of samples and the count of each class.

    """
    counts = df[label_col].value_counts()

    if counts.max() == len(df):
        leaf_class = counts.idxmax()
        print(f"{df_name}: Leaf node (class: {leaf_class})")
    else:
        print(f"{df_name}: Not a leaf node")
        print(f"Total points: {len(df)}")
        print("Class counts:")
        print(counts)




def plot_loan_3d(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = df["LoanDecision"].map({
        "Approve": "green",
        "Reject": "red"
    })

    ax.scatter(
        df["Income"],
        df["CreditScore"],
        df["CurrentDebt"],
        c=colors,
        s=20
    )

    ax.set_xlabel("Income")
    ax.set_ylabel("Credit Score")
    ax.set_zlabel("Current Debt")
    ax.set_title("Loan Decision 3D Plot")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label='Approve', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w',
               label='Reject', markerfacecolor='red', markersize=10)
    ]

    ax.legend(handles=legend_elements)

    plt.show()


  

# Read the CSV file
df = pd.read_csv("decision_tree.csv")


# Display the first few rows
print(df.head())
out1 = latex_df(df.head(), "Sample input data", "")       # copied to latex - has not impact on the script here 


# calculate the GI and S before we do anything.  
e_before = entropy(df["LoanDecision"])
print('Entropy before', e_before)
gi_before = gini(df["LoanDecision"])
print('Gini impurity before', gi_before)



# start branching
# Original features 
features = ["Income", "CreditScore", "CurrentDebt"]

# The first table help us decide which features we need to start splitting on.  
table = best_split_table(df, features)
print(table)

# This converts the df (table) to latex.
out2 = latex_df(df.head(), "Freture split comparison table", ""); print(out2)


# based on the table we decide to spliet on credit score first  
th1 = table[table['Feature'] == 'CreditScore']['BestGiniThreshold'].item()
plot_split(df, "CreditScore", th1)    # first figure 
fig = plt.gcf()          # Get current figure
fig.savefig("figure.png", dpi=300, bbox_inches="tight")



# next future - we impose the threshold on the original set and we have to choose a branch.
df_left = df[df['CreditScore'] <= th1]
df_right = df[df['CreditScore'] > th1]     # the branch we are staying on -- 

# This is credit score >  
features = ["Income", "CurrentDebt"]        # left features 
# what is the left split --- 
table1r = best_split_table(df_right, features, target="LoanDecision")
print(table1r)

# to do latex
out3 = latex_df(table1r , "Sample input data", ""); print(out3)    # so we can opy it to late.
# based on debt
th2d = table1r[table1r['Feature'] == 'CurrentDebt']['BestGiniThreshold'].item()


# Take a look at the figure 
plot_branch_2d(
    df=df,
    x_feature="CreditScore",
    y_feature="CurrentDebt",
    x_threshold=th1,                    # example credit score split
    y_threshold=th2d                    # example debt split
)
fig = plt.gcf()          # Get current figure
fig.savefig("second_split.png", dpi=300, bbox_inches="tight")



# This is credit score <  
features = ["Income", "CurrentDebt"]        # left features 
# what is the left split --- 
table1L = best_split_table(df_left, features, target="LoanDecision")
print(table1L)
th2i = table1L[table1L['Feature'] == 'Income']['BestGiniThreshold'].item()
# to do latex
out3 = latex_df(table1L , "Sample input data", ""); print(out3)  


# now we split it on income
plot_branch_2d(
    df=df,
    x_feature="CreditScore",
    y_feature="Income",
    x_threshold=th1,   # example credit score split
    y_threshold=th2i    # example debt split
)
fig = plt.gcf()          # Get current figure
fig.savefig("third_split.png", dpi=300, bbox_inches="tight")




# based on income -- plit the right side 
df_left_2nd = df_left[df_left['Income'] <= th2i]
df_right_2nd = df_left[df_left['Income'] > th2i]


e_before = entropy(df_left_2nd["LoanDecision"])
print('Entropy before', e_before)

gi_before = gini(df_left_2nd["LoanDecision"])
print('Gini impurity before', gi_before)

# Now we split the lower left quadrant 
features = ["CurrentDebt"]
table2L = best_split_table(df_left_2nd, features, target="LoanDecision")
print(table2L)
# to do latex 
latex_table = table2L.to_latex(
    index=False,
    caption="Pure Classification Rules",
    label="tab:pure_rules"
)
print(latex_table)

th2id = table2L[table2L['Feature'] == 'CurrentDebt']['BestGiniThreshold'].item()

# based on income -- plit the right side 
df_left_3nd = df_left[df_left['CurrentDebt'] <= th2id]
df_right_3nd = df_left[df_left['CurrentDebt'] > th2id]

# check for leave nodes?
check_leaf_node(df_left_3nd, 'df_left_3nd', "LoanDecision")
check_leaf_node(df_right_3nd, 'df_right_3nd' , "LoanDecision")






# my idea
features = ["Income", "CreditScore", "CurrentDebt"]

pure_rules = find_pure_thresholds(df, features)

print(pure_rules)

remaining_df = apply_pure_rules(df, pure_rules)

other_df = df.drop(remaining_df.index)

# Call the function
plot_loan_3d(df)

sys.exit() 

# check the other side
table2r = best_split_table(df_right_2nd, features, target="LoanDecision")
print(table2r)
# to do latex 
latex_table = table2r.to_latex(
    index=False,
    caption="Pure Classification Rules",
    label="tab:pure_rules"
)
print(latex_table)



# to do latex 
latex_table = pure_rules.to_latex(
    index=False,
    caption="Pure Classification Rules",
    label="tab:pure_rules"
)
print(latex_table)


remaining_df = apply_pure_rules(df, pure_rules)

other_df = df.drop(remaining_df.index)

# Call the function
plot_loan_3d(df)

# Done here 




table2r = best_split_table(df_right_2nd, features, target="LoanDecision")
print(table2r)
# to do latex 
latex_table = table2r.to_latex(
    index=False,
    caption="Pure Classification Rules",
    label="tab:pure_rules"
)
print(latex_table)







# split the left side
table2 = best_split_table(df_left, features, target="LoanDecision")
print(table2)

# to do latex 
latex_table = table2.to_latex(
    index=False,
    caption="Pure Classification Rules",
    label="tab:pure_rules"
)
print(latex_table)




# Now we got to go back to the left side and see if we can use the last feature to improve the results
df_left_2nd = df_left[df_left['Income'] <= th2i]
df_right_2nd = df_left[df_left['Income'] > th2i]
features = ["CurrentDebt"]
table3 = best_split_table(df_left_2nd, features, target="LoanDecision")
print(table3)
# based on debt
th3d = table3[table3['Feature'] == 'CurrentDebt']['BestGiniThreshold'].item()
# to do latex 
latex_table = table3.to_latex(
    index=False,
    caption="Pure Classification Rules",
    label="tab:pure_rules"
)
print(latex_table)







table2 = best_split_table(df_right_2nd, features, target="LoanDecision")
print(table1)






# based on debt
th2d = table2[table2['Feature'] == 'CurrentDebt']['BestGiniThreshold'].item()



plot_branch_2d(
    df=df,
    x_feature="CreditScore",
    y_feature="CurrentDebt",
    x_threshold=th1,                    # example credit score split
    y_threshold=th2d                    # example debt split
)


plot_branch_2d(
    df=df,
    x_feature="CreditScore",
    y_feature="Income",
    x_threshold=th1,   # example credit score split
    y_threshold=th2i    # example debt split
)


# Here we can still use one more split?
# next future 
df_left_2nd = df_left[df_left['CurrentDebt'] <= th2d]
df_right_2nd = df_left[df_left['CurrentDebt'] > th2d]
features = ["Income"]

table1 = best_split_table(df_left_2nd, features, target="LoanDecision")
print(table1)




# Here we can still use one more split?
# next future 
df_left_2nd = df_left[df_left['Income'] <= th2i]
df_right_2nd = df_left[df_left['Income'] > th2i]
features = ["CurrentDebt"]

table1 = best_split_table(df_left_2nd, features, target="LoanDecision")
print(table1)

table2 = best_split_table(df_right_2nd, features, target="LoanDecision")
print(table1)


# my idea
features = ["Income", "CreditScore", "CurrentDebt"]

pure_rules = find_pure_thresholds(df, features)

print(pure_rules)

remaining_df = apply_pure_rules(df, pure_rules)

other_df = df.drop(remaining_df.index)


# Display basic information
# print("\nData Information:")
# print(df.info())

# Display summary statistics
# print("\nSummary Statistics:")
# print(df.describe(include="all"))


