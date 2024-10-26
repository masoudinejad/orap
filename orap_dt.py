from sklearn.tree import DecisionTreeClassifier


class ORAPClassifier(DecisionTreeClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
