def load_stratified_dataset(path, labels, samples_per_label, random_seed=42):
    """
    Load from dataset, stratify labels
    Inputs: path (path to dataset)
            labels (name of column to stratify by)
            samples_per_label (number of samples per label)
            random_seed: (to get articles, default is 42)
    Output: dataframe with even distribution of labels
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    # Split dataset, each category with same sample size
    l = []
    df = pd.read_csv(path)
    for label in df[labels].unique():
        tmp = df[df[labels] == label]
        # Check if label has enough samples
        if samples_per_label > tmp.shape[0]:
            # Get smallest label
            smallest = tmp.shape[0]
            for label in df[labels].unique():
                num = df[df[labels] == label].shape[0]
                if num < smallest:
                    smallest = num
            print("Smallest sample size in dataset is {} samples!".format(smallest))
            # Load dataset with smallest label
            return load_stratified_dataset(path, labels, smallest)
        # Get smaller dataset, append it to new one
        tmp, _ = train_test_split(tmp, shuffle=True, 
                                  test_size=abs(1-samples_per_label/(tmp.shape[0]-0.001)),
                                  random_state=random_seed)
        l.append(tmp)
    df = pd.concat(l, axis=0, ignore_index=True)
    
    return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)