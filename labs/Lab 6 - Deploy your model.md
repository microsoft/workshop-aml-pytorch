# Lab 6 - Deploy your model


### Connect with your environment
```
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)

# get details of the current Azure ML workspace
ws = Workspace.from_config()

# default authentication flow for Azure applications
default_azure_credential = DefaultAzureCredential()

# client class to interact with Azure ML services and resources, e.g. workspaces, jobs, models and so on.
ml_client = MLClient(
   default_azure_credential,
   ws.subscription_id,
   ws.resource_group,
   ws.name
)
```

### Create an endpoint
This can take around 2 minutes
```
online_endpoint_name = "endpoint-lego"

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="This is a endpoint that can classify Lego Characters",
    auth_mode="key"
)

ml_client.begin_create_or_update(endpoint)
```

### Create a deployment in your endpoint 
This can take around 15 minutes
```
model = ml_client.models.get(name="LegoCharacters", label="latest")

env = Environment(
    image="mcr.microsoft.com/azureml/pytorch-1.10-ubuntu18.04-py37-cpu-inference:20220516.v3"
)

deployment  = ManagedOnlineDeployment(
    name="version-1",
    endpoint_name=online_endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./src", 
        scoring_script="score.py"
    ),
    instance_type="Standard_F2s_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment)
```

### Test the deployment
```
```
