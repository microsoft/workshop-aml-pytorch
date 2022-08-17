# Lab 4 - Training in the cloud

**In this lab you are going to move the training of the model you created in Lab 3 to the Azure Machine Learning and train it on a GPU enabled compute cluster.**

TODO: Insert create compute


## Quick try
Let's start with a "Hello World" example to see if everything is working as expected. 

Create a new Notebook and add the the script below and run the cell. The output of the job is streamed directly to the notebook. When the job is done you should see Hello world.

```
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace

# get details of the current Azure ML workspace
ws = Workspace.from_config()

# default authentication flow for Azure applications
default_azure_credential = DefaultAzureCredential()
subscription_id = ws.subscription_id
resource_group = ws.resource_group
workspace = ws.name

# client class to interact with Azure ML services and resources, e.g. workspaces, jobs, models and so on.
ml_client = MLClient(
   default_azure_credential,
   subscription_id,
   resource_group,
   workspace)

# target name of compute where job will be executed
computeName="gpu-cluster"
job = command(
    code="./src",
    command="python hello.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute=computeName,
    display_name="hello-world-example",
)

returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
ml_client.jobs.stream(returned_job.name)
```

## Move the data to the cloud

Now that we have our cluster up and running we need to move our data from the compute target to the cloud. We create a dataset from it and version it. Now we can always trace back our date.

Copy and paste the code below in a cell and run the cell.

```
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = './data/PetImages'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="Cat and Dogs dataset",
    name="Cats-and-Dogs",
    version="1"
)

ml_client.data.create_or_update(my_data)
```


## Create a training job

Copy: ../scr/train.py

``` 
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from azure.ai.ml.entities import Data, Model
from azure.ai.ml.constants import AssetTypes
from azureml.core import Workspace

# get details of the current Azure ML workspace
ws = Workspace.from_config()

# default authentication flow for Azure applications
default_azure_credential = DefaultAzureCredential()
subscription_id = ws.subscription_id
resource_group = ws.resource_group
workspace = ws.name

# client class to interact with Azure ML services and resources, e.g. workspaces, jobs, models and so on.
ml_client = MLClient(
    default_azure_credential,
    subscription_id,
    resource_group,
    workspace)

# the key here should match the key passed to the command
my_job_inputs = {
    "data_path": Input(type=AssetTypes.URI_FOLDER, path="azureml:Cats-and-Docs:3")
}

# target name of compute where job will be executed
computeName="gpu-cluster"
job = command(
    code="./src",
    # the parameter will match the training script argument name
    # inputs.data_path key should match the dictionary key
    command="python train.py --data_path ${{inputs.data_path}} --num_epochs 1 --dataset_size='small' --model_output_path ${{outputs.model}}",
    inputs=my_job_inputs,
    outputs=dict(model=Output(type=AssetTypes.CUSTOM_MODEL)),
    environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest",
    compute=computeName,
    display_name="day1-experiment-data",
)

returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)

ml_client.jobs.stream(returned_job.name)
```

## Register the model

```
model_path = f"azureml://jobs/{returned_job.name}/outputs/model"

model = Model(name="CatsAndDogs",
                path=model_path,
                type=AssetTypes.CUSTOM_MODEL)
registered_model = ml_client.models.create_or_update(model)
```