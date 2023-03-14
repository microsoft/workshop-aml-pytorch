# Lab 5 - Deploy your model

In this lab we are going to take the model you created in Lab 4 and deploy it in a manage endpoint. When the model is deployed in an managed enpoint you can call the endpoint point with a REST call.   
   
Continue in the same notebook as for Lab 4.


### Create an endpoint
This can take around 2 minutes
```
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)

online_endpoint_name = "endpoint-lego"

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = online_endpoint_name,
    description = "This is a endpoint that can classify Lego Characters",
    auth_mode = "key"
)

ml_client.begin_create_or_update(endpoint)
```

### Create a deployment in your endpoint 

First we download the scoring and the environment files. 
The scoring file in the endpoint handles the inference in the endpoint and the environment file 
contains the conda packages needed to run the scoring file.

```
!wget https://raw.githubusercontent.com/microsoft/workshop-aml-pytorch/main/src/score.py -P src
!wget https://raw.githubusercontent.com/microsoft/workshop-aml-pytorch/main/src/environment.yml -P src
``` 

Create an custom enviroment that is optimized for inference for our model.

> Azure Machine Learning environments define the execution environments for your jobs or deployments and encapsulate the dependencies for your code. Azure Machine Learning uses the environment specification to create the Docker container that your training or scoring code runs in on the specified compute target. You can define an environment from a conda specification, Docker image, or Docker build context.

```
env_docker_conda = Environment(
    image = "mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest",
    conda_file = "src/environment.yml",
    name = "lego-characters-inference-cpu",
    description = "Environment",
)

ml_client.environments.create_or_update(env_docker_conda)
```

Deployments can take around 15 minutes

```
# Get the latest model from the Model Repository
model = ml_client.models.get(name="LegoCharacters", label="latest")

# Specify the inference environment 
env = ml_client.environments.get(name="lego-characters-inference-cpu",label="latest")

deployment  = ManagedOnlineDeployment(
    name = "version-1",
    endpoint_name = online_endpoint_name,
    model = model,
    environment = env,
    code_configuration = CodeConfiguration(
        code = "./src", 
        scoring_script = "score.py"
    ),
    instance_type = "Standard_F2s_v2",
    instance_count = 1,
)

ml_client.online_deployments.begin_create_or_update(deployment)
```

### Update the traffic
```
endpoint.traffic = {"version-1": 100}
ml_client.begin_create_or_update(endpoint)
```

### Test the deployment
```
!wget https://raw.githubusercontent.com/microsoft/workshop-aml-pytorch/main/src/sample-request.json -P src

ml_client.online_endpoints.invoke(
    endpoint_name = online_endpoint_name,
    deployment_name = "version-1",
    request_file = "./src/sample-request.json",
)
```


( --- )

You now has succesfully trained a PyTorch classification model and deployed the model in a managed endpoint.



### More resources:
- [Microsoft Docs](https://learn.microsoft.com/azure/machine-learning/how-to-safely-rollout-managed-endpoints-sdk-v2)
