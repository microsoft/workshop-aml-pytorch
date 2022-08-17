# Lab 1 - Set an Azure Machine Learning Workspace

**In this lab, you'll create a workspace and then add compute resources to the workspace. You'll then have everything you need to get started with Azure Machine Learning.**

### Prerequisites
An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=aiml-73287-heboelma).

>**What is a workspace?**   
>The workspace is the top-level resource for your machine learning activities, providing a centralized place to view and manage the artifacts you create when you use Azure Machine Learning. The compute resources provide a pre-configured cloud-based environment you can use to train, deploy, automate, manage, and track machine learning models.


## Create a Azure Machine Learning Workspace

- Sign in to the [Azure portal](https://portal.azure.com?WT.mc_id=aiml-73287-heboelma) by using the credentials for your Azure subscription.
- To create Azure Machine Learning resource [click here](https://portal.azure.com/#create/Microsoft.MachineLearningServices?WT.mc_id=aiml-73287-heboelma)
- Provide the following information to configure your new workspace:   
  
| Field	| Description |
| --- | --- |
| Workspace name | Enter a unique name that identifies your workspace. In this example, we use amlpt-ws. Names must be unique across the resource group. Use a name that's easy to recall and to differentiate from workspaces created by others. | 
| Subscription | Select the Azure subscription that you want to use. |
Resource group|	Use an existing resource group in your subscription, or enter a name to create a new resource group. A resource group holds related resources for an Azure solution. In this example, we use amlpt-aml. |
| Region |	Select the location closest to your users and the data resources to create your workspace. |
Storage account |	A storage account is used as the default datastore for the workspace. You may create a new Azure Storage resource or select an existing one in your subscription. |
| Key vault |A key vault is used to store secrets and other sensitive information that is needed by the workspace. You may create a new Azure Key Vault resource or select an existing one in your subscription.|
| Application insights |The workspace uses Azure Application Insights to store monitoring information about your deployed models. You may create a new Azure Application Insights resource or select an existing one in your subscription. |
| Container registry |	A container registry is used to register docker images used in training and deployments. You may choose to create a resource or select an existing one in your subscription. |
- After you're finished configuring the workspace, select **Review + Create**.
- Select Create to create the workspace.
  > It can take several minutes to create your workspace in the cloud.
- When the process is finished, a deployment success message appears.
- To view the new workspace, select Go to resource.
- From the portal view of your workspace, select **Launch studio** to go to the Azure Machine Learning studio.

## Create a compute instance
You could install Azure Machine Learning on your own computer. But in this workshop, you'll create an online compute resource that has a development environment already installed and ready to go. You'll use this online machine, a compute instance, for your development environment to write and run code in Python scripts and Jupyter notebooks.

Create a compute instance to use this development environment for the rest of the tutorials and quickstarts.

- On the left side, select Compute.
- Select **+New** to create a new compute instance.
- Supply a name
- Select CPU and select "Standard_DS3_v2" as virtual machine size
- Select **Create**.
- In about two minutes, you'll see the State of the compute instance change from Creating to Running. It's now ready to go.




## Learn more on Microsoft Docs
[Quickstart: Create workspace resources you need to get started with Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-73287-heboelma)
[Read more on Docs](https://docs.microsoft.com/azure/machine-learning/?WT.mc_id=aiml-73287-heboelma)


## Review

Your development environment in your Azure Machine Learning environment is now ready to use. You can continue to the next lab.      
In this lab you have:

* [ ] Created an Azure Machine Learning Workspace
* [ ] Created a Compute Instance in your Azure Machine Learning Workspace