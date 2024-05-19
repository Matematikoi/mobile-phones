import file_management as fm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer


class CellPhoneModel:
    available_models = [
        "LOGISTIC_REGRESSION",
        "SVC",
        "DECISION_TREE",
        "RANDOM_FOREST",
        "GRADIENT_BOOSTING",
    ]

    def __init__(self, model_name, params):
        """
        An instance of the CellPhoneModel will perform a CV hyperparameter
        tunning using the training dataset.

        We pass the model we want to use as model, and the parameters we want to use as
        params
        """
        if model_name not in self.available_models:
            raise Exception(
                f'Model "{model_name}" not implemented, available models are {self.available_models}'
            )
        self.model_name = model_name
        self.params = params

    def get_data(self):
        """
        Get the data using file_management.pu
        """
        # Get Data
        target = "price_range"
        data = fm.read_parquet(fm.Filenames.train)
        X = data.drop(columns=target)
        y = data[target]

        X_val_train, self.X_test, y_val_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_val_train, y_val_train, test_size=0.3, random_state=42
        )

    def train(self):
        """
        Trains the model and saves the model data
        """
        self.get_data()

        pipeline_params = [StandardScaler()]
        # Switch for the model
        if self.model_name == "LOGISTIC_REGRESSION":
            pipeline_params.append(LogisticRegression(random_state=42))
        elif self.model_name == "SVC":
            pipeline_params.append(SVC(random_state=42))
        elif self.model_name == "DECISION_TREE":
            pipeline_params.append(DecisionTreeClassifier(random_state=42))
        elif self.model_name == "RANDOM_FOREST":
            pipeline_params.append(RandomForestClassifier(random_state=42))
        elif self.model_name == "GRADIENT_BOOSTING":
            pipeline_params.append(GradientBoostingClassifier(random_state=42))

        model = make_pipeline(*pipeline_params)
        f1 = make_scorer(f1_score, average="micro")

        model_grid = GridSearchCV(
            model, param_grid=self.params, cv=5, n_jobs=-1, verbose=1, scoring=f1
        )
        model_grid.fit(self.X_train, self.y_train)
        self.model = model_grid

    def get_score_val(self):
        """
        Get the f1 micro result for validation
        """
        return self.model.score(self.X_val, self.y_val)

    def predict(self, data):
        """
        Given a matrix X it predict a vector y
        """
        self.model.predict(data)


def main():
    model_params = [
        (
            "LOGISTIC_REGRESSION",
            {
                "logisticregression__C": [0.001, 0.1, 0.5, 1.0],
                "logisticregression__solver": [
                    "lbfgs",
                    "newton-cg",
                    "newton-cholesky",
                    "sag",
                    "saga",
                ],
                "logisticregression__penalty": ["l1", "l2", "elasticnet"],
                "logisticregression__max_iter": [100, 10000],
            },
        ),
        (
            "SVC",
            {
                "svc__C": [0.001, 0.1, 0.5, 1.0],
                "svc__gamma": [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
                "svc__kernel": ["rbf", "linear", "poly", "sigmoid", "precomputed"],
            },
        ),
        (
            "DECISION_TREE",
            {
                "decisiontreeclassifier__criterion": ["gini", "entropy"],
                "decisiontreeclassifier__splitter": ["best", "random"],
                "decisiontreeclassifier__max_depth": [3, 5, 7, 10],
                "decisiontreeclassifier__min_samples_split": range(2, 11, 1),
                "decisiontreeclassifier__min_samples_leaf": range(2, 10, 1),
            },
        ),
        (
            "RANDOM_FOREST",
            {
                "randomforestclassifier__n_estimators": range(25, 151, 50),
                "randomforestclassifier__criterion": ["gini", "entropy"],
                "randomforestclassifier__max_depth": [5, 7, 10],
                "randomforestclassifier__min_samples_split": range(2, 100, 30),
                "randomforestclassifier__min_samples_leaf": range(2, 100, 30),
            },
        ),
        (
            "GRADIENT_BOOSTING",
            {
                "gradientboostingclassifier__n_estimators": range(25, 151, 25),
                "gradientboostingclassifier__max_depth": [3, 5, 7, 10],
            },
        ),
    ]

    result = []
    for model, param in model_params:
        print(f"Training model {model}")
        m = CellPhoneModel(model, param)
        m.train()
        result.append(m)


if __name__ == "__main__":
    main()
