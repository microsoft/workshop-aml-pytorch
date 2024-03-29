# Lab 4 - Training in the cloud

**In this lab you are going to move the training of the model you created in Lab 3 to the Azure Machine Learning and train it on a GPU enabled compute cluster.**

## 4.1 - A quick try
Let's start with a "Hello World" example to see if everything is working as expected. 

Create a new Notebook and add the the script below and run the cell. The output of the job is streamed directly to the notebook. When the job is done you should see Hello world.

#### Create a dummy training script

We start with creating a dummy training script called "hello.py" and put it in the folder "src" relatively from your notebook.

```
!mkdir src2
!echo "print(\"hello world\")" > "src2/hello.py"
```

#### Connect to the Azure ML Workspace
Now that we have the training file, we start with connecting to our Azure ML Workspace. 

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

print("Connected to: "+ workspace)
```

#### Create a compute resource for training
An AzureML compute cluster is a fully managed compute resource that can be used to run the training job. In the following examples, a compute cluster named gpu-cluster is created.

```
from azure.ai.ml.entities import AmlCompute

# specify aml compute name.
cpu_compute_target = "gpu-cluster"

try:
    ml_client.compute.get(gpu)
except Exception:
    print("Creating a new gpu compute target...")
    compute = AmlCompute(
        name=cpu_compute_target, size="Standard_NC6", min_instances=0, max_instances=1
    )
    ml_client.compute.begin_create_or_update(compute).result()
```

#### Submit the training job
```
# target name of compute where job will be executed
computeName="gpu-cluster"
job = command(
    code="./src2",
    command="python hello.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute=computeName,
    display_name="hello-world-example",
)

returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)

print("The outputs of the job:")
ml_client.jobs.stream(returned_job.name)
```

If the job is completed you should see "Hello World" in the outputs from the job. If it does not appear in the output, follow the link displayed in the output. Go to the tab: "Outputs + Logs" open de folder "user_logs" and select the file: "std_log.txt".


## 4.2 Move the data to the cloud

Now that we have our cluster up and running we need to move our data from the compute target to the cloud. We create a dataset from it and version it. Now we can always trace back our date.

Copy and paste the code below in a cell and run the cell to create a dataset from the data

```
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = './data/dataset-lego-characters/dataset'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="Lego Characters Dataset",
    name="Lego-Characters",
    version="1"
)

ml_client.data.create_or_update(my_data)
```


## 4.2 Create the training job

We start with downloading the training file from the repository. In this file is all the code from Lab 3.

``` 
!mkdir src
!wget https://raw.githubusercontent.com/microsoft/workshop-aml-pytorch/main/src/train.py -P src
``` 

Now lets create the training job on our compute cluster using the dataset.

``` 
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from azure.ai.ml.entities import Data, Model
from azure.ai.ml.constants import AssetTypes
from azureml.core import Workspace

# the key here should match the key passed to the command
job_inputs = {
    "data_path": Input(type=AssetTypes.URI_FOLDER, path="azureml:Lego-Characters:1"),
    "num_epochs": 4
}

job_outputs = {
    "model":Output(type=AssetTypes.CUSTOM_MODEL)
}

# target name of compute where job will be executed
computeName = "gpu-cluster"

job = command(
    inputs = job_inputs,
    outputs = job_outputs,
    code = "./src",
    command = "python train.py --data_path ${{inputs.data_path}} --num_epochs ${{inputs.num_epochs}} --model_output_path ${{outputs.model}}",
    environment = "AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest",
    compute = computeName,
    display_name="Lego Characters Model Traning",
)

returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)

ml_client.jobs.stream(returned_job.name)
```


## Register the model
Now that the training job is done, we can register the model in Model Management. When the model is in Model Management we can use it in deployments.

```
model_path = f"azureml://jobs/{returned_job.name}/outputs/model"

model = Model(name="LegoCharacters",
                path=model_path,
                type=AssetTypes.CUSTOM_MODEL)
registered_model = ml_client.models.create_or_update(model)

print("Model version: "+registered_model.version)
```

Let's move on to the next lab and deploy the model in an endpoint so we can use it in a solution.

( --- )

[Continue with: Lab 5 - Deploy your model](./Lab%205%20-%20Deploy%20your%20model.md)