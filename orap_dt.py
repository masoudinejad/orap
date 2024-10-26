from sklearn.tree import DecisionTreeClassifier


class ORAPClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
    ) -> None:
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )

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
