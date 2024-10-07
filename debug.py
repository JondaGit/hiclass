from sklearn.ensemble import RandomForestClassifier
from hiclass.ConstantClassifier import ConstantClassifier
from hiclass.HierarchicalClassifier import HierarchicalClassifier
from hiclass.LocalClassifierPerParentNode import LocalClassifierPerParentNode


from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Creating a synthetic dataset with 3 levels of hierarchy
X, y_level_1 = make_classification(
    n_samples=100, n_features=20, n_informative=10, n_classes=3, random_state=42
)

# Define hierarchical levels
y_level_2 = np.zeros_like(y_level_1)
y_level_3 = np.zeros_like(y_level_1)

# Example hierarchy: Level 1: A, B, C; Level 2: A1, A2, B1, B2, C1, C2; Level 3: A1a, A1b, etc.
for i in range(len(y_level_1)):
    if y_level_1[i] == 0:
        y_level_2[i] = np.random.choice([0, 1])  # A1, A2
        if y_level_2[i] == 0:
            y_level_3[i] = np.random.choice([0, 1])  # A1a, A1b
        else:
            y_level_3[i] = np.random.choice([2, 3])  # A2a, A2b
    elif y_level_1[i] == 1:
        y_level_2[i] = np.random.choice([2, 3])  # B1, B2
        if y_level_2[i] == 2:
            y_level_3[i] = np.random.choice([4, 5])  # B1a, B1b
        else:
            y_level_3[i] = np.random.choice([6, 7])  # B2a, B2b
    elif y_level_1[i] == 2:
        y_level_2[i] = np.random.choice([4, 5])  # C1, C2
        if y_level_2[i] == 4:
            y_level_3[i] = np.random.choice([8, 9])  # C1a, C1b
        else:
            y_level_3[i] = np.random.choice([10, 11])  # C2a, C2b


y_hierarchical = np.vstack((y_level_1, y_level_2, y_level_3)).T

X_train, X_test, y_train, y_test = train_test_split(
    X, y_hierarchical, test_size=0.3, random_state=42
)

print(X_train[:5], y_train[:5])


clf = RandomForestClassifier(random_state=42)

hclf = LocalClassifierPerParentNode(clf, replace_classifiers=True)
hclf.fit(X_train, y_train)

hclf.predict_proba(X_test[:3])
