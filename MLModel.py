import numpy as np
import pandas as pd

df = pd.read_csv("https://workshopst.blob.core.windows.net/data-worshop/IRIS.csv")
df.head(5)

df.info()

df['species'].value_counts()

import matplotlib.pyplot as plt

plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = df
test_train_split_ratio = 0.3

train_df, test_df = train_test_split(data, test_size=test_train_split_ratio)

y_train = train_df.pop('species')
y_test = test_df.pop('species')

x_train =train_df.values
x_test = test_df.values

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

print('accuracy = ', model.score(x_test, y_test))

model.predict([[5.1, 10, 5, 3]])

import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--test_train_ratio', type=float)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--registered_name', type=str)
    args = parser.parse_args()

    mlflow.start_run()
    mlflow.sklearn.autolog()


    data = pd.read_csv(args.data)
    test_train_split_ratio = args.test_train_ratio

    train_df, test_df = train_test_split(data, test_size=test_train_split_ratio)

    mlflow.log_metric('num_rows', data.shape[0])

    y_train = train_df.pop('species')
    y_test = test_df.pop('species')

    x_train =train_df.values
    x_test = test_df.values

    model = KNeighborsClassifier(n_neighbors=args.k)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    print('accuracy = ', model.score(x_test, y_test))

    mlflow.sklearn.log_model(sk_model=model,   
                registered_model_name=args.registered_name,
                artifact_path = args.registered_name)

    mlflow.sklearn.save_model(
        sk_model = model,
        path=os.path.join(args.registered_name, "trained_model")
    )

    mlflow.end_run()

if __name__ = "__main__":
    main()

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="3ae21168-ac7b-47ee-9837-85bc80dd07b3",
    resource_group_name="savci.gizem02-rg",
    workspace_name="workshop-tutorial"
)

from azure.ai.ml import command
from azure.ai.ml import Input

registered_name = "iris_class_model"

job = command(
    inputs = dict(
        data = Input(
            type="uri_file",
            path = "https://workshopst.blob.core.windows.net/data-worshop/IRIS.csv"
        ),
        test_train_ratio=0.2,
        k=2,
        registered_name = registered_name
    ),
    code="./src/",  # location of source code
    command="python main.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --k ${{inputs.k}} --registered_model_name ${{inputs.registered_model_name}}",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    # compute="cpu-cluster"
    # if (cpu_cluster)
    # else None,  # No compute needs to be passed to use serverless
    display_name="iris_class_model_prediction",
)

# submit job
ml_client.create_or_update(job)

