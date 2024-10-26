from sklearn.tree import DecisionTreeClassifier


class ORAPClassifier(DecisionTreeClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_depths(self) -> list[int]:
        node_count = self.tree_.node_count
        parent_count = [0] * node_count
        for i in range(1, node_count):
            parent = (
                self.tree_.children_left[i]
                if self.tree_.children_left[i] != -1
                else self.tree_.children_right[i]
            )
            parent_count[i] = parent_count[parent] + 1
        self.depths = parent_count
