# Deploying a Model with AWS SageMaker

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks and Pytorch Guide
3. Convolutional Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model with AWS SageMaker

Each module has a folder with its respective notes. This folder is the one of the **sixth module**: Deployment.

Additionally, note that:

- I made many hand-written notes; check the PDFs.
- I forked the Udacity repositories for the exercises; most the material and notebooks are there:
  - [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)
  - [DL_PyTorch](https://github.com/mxagar/DL_PyTorch)
  - [sagemaker-deployment](https://github.com/mxagar/sagemer-deployment)
- If you are interested on my notes on Google Colab, check: [`Google_Colab_Notes.md`](https://github.com/mxagar/computer_vision_udacity/blob/main/02_Cloud_Computing/Google_Colab_Notes.md).

## Overview of Contents

- [Geneartive Adversarial Networks (GAN)](#geneartive-adversarial-networks-gan)
  - [Overview of Contents](#overview-of-contents)
  - [1. Introduction to Deployment](#1-introduction-to-deployment)
  - [2. Building a Model Using SageMaker](#2-building-a-model-using-sagemaker)
  - [3. Deploying and Using a Model](#3-deploying-and-using-a-model)
  - [4. Hyperparamter Tuning](#4-hyperparamter-tuning)
  - [5. Updating a Model](#5-updating-a-model)
  - [6. Project: Deploying a Sentiment Analysis Model](#6-project-deploying-a-sentiment-analysis-model)

## 1. Introduction to Deployment

Deployment in the Cloud is the focus of this module, concentrating on AWS SageMaker. However, the concepts are valid for any cloud platform.

Learned questions:

- What's the machine learning workflow?
- How does deployment fit into the machine learning workflow?
- What is cloud computing?
- Why would we use cloud computing for deploying machine learning models?
- Why isn't deployment a part of many machine learning curriculums?
- What does it mean for a model to be deployed?
- What are the essential characteristics associated with the code of deployed models?
- What are different cloud computing platforms we might use to deploy our machine learning models?

### 1.1 Machine Learning Workflow

The general machine learning workflow has these primary components with their sub-steps:

1. Explore & Process
  - Retrieve data
  - Clean data
  - Explore data
  - Prepare/transform
  - Split: train/validation/test
2. Modeling
  - Develop model
  - Train
  - Validate: tune, select best model
  - Evaluate model: test split
3. Deployment: **we focus here**
  - Deploy to production
  - Monitor
  - Update

Note this is **cyclical**! We start again when we see we need to update out model!

![Machine Learning Workflow](./pics/ml_workflow.jpg)

The third component, **deployment**, is where the section and the module focus on. Note that in the personal and the academic environment deployment is not relevant &mdash; but in the work environment is!

The different cloud providers describe their machine learning workflow as follows:

- [Machine Learning with Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-mlconcepts.html)
- [Machine learning workflow on GCloud](https://cloud.google.com/ai-platform/docs/ml-solutions-overview)
- [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning)


### 1.2 Cloud Computing

An abstract definition of cloud computing: convert an IT product into an IT service; e.g., USB stick becomes GDrive.

The nice thing of a cloud service is that its capacity scales with the demand; that doesn't happen with traditional infrastructure.

![Cloud Services: Capacity Scales](./pics/cloud_capacity.png)

Note that the capacity can be understood as the number of IT resources (storage, compute, network, etc.), and it has a cost associated to it.

Ideally we want to follow the black demand curve, otherwise, with a traditional infrastructure we have either:

- wasted capacity (area below the blue curve bounded by the black)
- or insufficient capacity (area below the black curve bounded by the blue).

That is clearly a missuse of resources: we are either loosing customers or money, i.e., in any case we're always loosing money.

With cloud infrastructures we can follow the demand (registered users) and trigger automatically increased/decreased capacity. The area between the black and the yellow curves is the dynamic capacity.

#### Pros and Cons

**Benefits** of cloud computing:

- Reduced investments, proportional costs: we don't need to buy and maintain servers, but we use them and pay proportionally to our usage.
- Scalability, better capacity planning: automatic triggers allocate more resources depending on users registered, i.e., demand.
- Increased availability and reliability (thus, organizational agility).

But cloud computing has also **risks**:

- (Potential) Increase in Security Vulnerabilities
- Reduced Operational Governance Control (over cloud resources)
- Limited Portability Between Cloud Providers
- Multi-regional Compliance and Legal Issues

> Indeed, the **Service Level Agreements (SLA)** provided for a cloud service often highlight security responsibilities of the cloud provider and **those assumed by the cloud user**.

In other words, the cloud providers assume responsibilities of the user regarding the risk surface they have.

More on [AWS Security](https://aws.amazon.com/security/security-learning/?cards-top.sort-by=item.additionalFields.sortDate&cards-top.sort-order=desc&awsf.Types=*all).

#### Definitions (NIST)

The National Institute of Standards and Technology (NIST) defined in 2011 cloud computing using 

1. service models,
2. deployment models,
3. and essential characteristics

as shown in the following image:

![NIST: Definition of Cloud Computing](./pics/nist_cloud_computing.png)

Since the, each cloud provider updated their definition, but we can take the NIST definition as reference.

There are three **software service models** depending on 

- which **cloud components** they comprise
- how the **responsibility** is delegated between the cloud provider and the customer.

![NIST: Software Services](./pics/nist_cloud_computing_service_models_saas.png)

The **service model** examples are:

- Software as a Service (SaaS): Google Docs, GMail; as opposed to *software as a product*, in SaaS the application is on the cloud and we access it via browser. The user has the unique responsibility of the login and the administration of the application and the content.
- Platform as a Service (PaaS): Heroku; we can use PaaS to e-commerce websites, deploy an app which is reachable via web or a REST API, etc. usually, easy deployments at the application level are done. Obviously, the user that deploys the application has more responsibilities.
- Infrastructure as a Service (IaaS): AWS; they offer virtual machines on which the user needs to do everything: virtual machine provisioning, networking, app deployment, etc.

The **deployment models** are distinguished by the group for which the service is being provided:

- Public: for use by the general public; AWS, GCloud, Azure, etc. They are the least secure, but they also enable virtual private clouds.
- Community: Government Clouds; they are more secure, because of restricted access.
- Private and Hybrid Clouds: Company clouds, with servers in the company.

The **essential characteristics** are 

- On-Demand Self Service: no human interaction, customer performs automatic provisioning.
- Broad Network Access: we can access from any device with internet.
- Resource Pooling: many customers with very different requirements need to be served.
- Rapid Elasticity: scaling of compute capabilities depending on demand.
- Measured Service: the cloud provider automatically controls and optimizes resource usage.

#### Cloud Computing Guidelines

Cloud computing is perfect for **start ups**, because

- They don't have infrastructure overhead costs, they pay as they go.
- It requires fewer staff.
- It can scale.
- It enables placing the product to market faster.

For established companies, cloud computing is not always the way to go, because they might have legacy architecture and their staff lacks the skillset.

Successful examples:

- [Instagram](https://instagram-engineering.com/migrating-from-aws-to-aws-f4b16a65e13c), which started from scratch at AWS in 2010. They migrated to Facebook serves after their purchas in 2012: [Migrating From AWS to FB](https://instagram-engineering.com/migrating-from-aws-to-fb-86b16f6766e2).
- [Netflix](https://aws.amazon.com/solutions/case-studies/netflix/) migrated from using its own servers in 2009 to AWS in 2010.

#### Cloud Computing within the Machine Learning Workflow

Depending on the amount of on-premise infratructure available at our organization and the amount of risk associated to the cloud technology we'd like to face, we can choose to:

- Implement all three components on-premises
- Implement all three components on the cloud
- Implement any of the last two components on the cloud: modeling and/or deployment.

Amazon SageMaker allows for having all 3 components on the cloud.

It is also quite common to have only the deployment component on the cloud due to security reasons.

![ML Workflow: All in Cloud vs None](mlworkflow-allornocloud.png)

![ML Workflow: Parts in Cloud](mlworkflow-differentcloudservices.png)

### 1.3 Paths to Deployment

The most common ways to deploy a machine learning model have been:

1. Recode the python model into C++/java
2. Recode the model into Predictive Model Markup Language (PMML) or Portable Format Analytics (PFA).
3. **Convert python model into a format used in the production environment.** This format can be a binary or code that is compiled; the keyword is **convert**, i.e., we don't recode anything.

In recent years, the last way has become popular and seems to be the future. It's the easiest and fastest way and many frameworks (Scikit-Learn, Pytorch, etc.) are already able to do that. Sometimes intermediate formats are used, such as [ONNX](https://onnx.ai/).

Another aspect in the deployment process is who does it. Traditionally, deployment has belonged to operations and software/platform/DevOps engineers were in charge of it. However, in recent years data scientists/analysts/ML engineers have started to be responsible for it because tools for easy deployment have appeared or evolved, such as:

- Containers
- Tools for creating REST APIs
- AWS SageMaker
- Google ML Engine / [Google Vertex AI](https://cloud.google.com/vertex-ai) (equivalent to SageMaker)

![Machine Learning Workflow and DevOps](./pics/mlworkflow-devops.png)

### 1.4 Production and Test Environments

Usually, machine learning applications are deployed with the following architecture:

![Production Environment](./pics/production_environment.jpg)

We have these parts:

- The users, who input data and get predictions associated with that data.
- The application, which is the interface to the users and the model; the application is in the **production environment**.
- The model, which usually is not in the application, but its interfaced by it. The interface between the application and the model happens in the so called **end points**, which get the user data and provide the prediction.

Note that instead of the production environment we can have a **test environment** if we are performing tests, i.e., there's no real user, but a tester (person or bot). A **production environment** is characterized by the fact that it's being used by real users.

Thus, the **type of environment (test/production)** is determined by who uses the service.

### 1.5 Endpoints and REST APIs

One way of understanding endpoints is the following:

- the ENDPOINT itself is like a function call
- the function itself would be the model and
- the Python program is the APPLICATION.

```python
# APPLICATION = The python program/script
def main():
    input_user_data = get_user_data()
    # ENDPOINT = Function call
    predictions = ml_model(input_user_data)
    display_predictions_to_user(predictions)

def ml_model(user_data):
    loaded_data = load_user_data(user_data)
    # ...
    return predictions
```

Often, the connection to the endpoint is done using a **REST API**: REpresentational State Transfer Application Programming Interface. Basically, we have a service in which the model is contained and that service is able to

- receive a **HTTP request**,
- process the request and feed it to the model,
- package the model output,
- and send a **HTTP responses** which contains the model output.

An **HTTP request** has four parts:

1. Endpoint: that's the URL which targets a specific function.
2. HTTP Method: any of these four (CRUD):
    - GET: Read
    - POST: Create (usually that's the one when we're trying to send data to get a prediction)
    - PUT: Update
    - DELETE: Delete
3. HTTP Headers: data format in the message, additional info, etc.
4. Message: the input data sent by the user.

![HTTP Methods](./pics/http_methods.png)

An **HTTP response** has three parts:

1. HTTP Status Code: if data successfully received, code should start with 2, e.g., 200.
2. HTTP Headers: data format in the message, additional info, etc.
3. Message: the output data sent to the user, i.e., the prediction.

It is the application's responsibility to format the input/output data correctly for/from the model interfacing with the user. Usually, the data is formatted in CSV/JSON format.


### 1.6 Containers

The model and the application need a computing environment; often, that computing environment is a **container** (one for each). Docker is the most popular container technology.

A container is an isolated computational environment which contains all the libraries and software necessary to run an application or the part of an application.

A container can be mistaken with a virtual machine (VM), but it's not a VM, because it uses the resources of the underlying operating system via the container engine. However, for instance, we can run Linux-based container on Mac/Windows. Since they're not virtual machines, they're much lighter.

![Containers](./pics/containers.png)

Containers are defined in image scripts which specify in layers the software components that build the container.

In Docker:

- Images are Dockerfile scripts
- Built/instantiated images are containers
- DockerHub is an image registry, i.e., a repository where container images are hosted.

Advantages of containers:

- Application is isolated, i.e., more secure.
- Requires only software to run the application.
- Application creation, replication, sharing, deletion is easier.
- We package the application in a container and it runs everywhere!

Udacity workspaces run on containers.

### 1.7 Characteristics of Deployment and Modeling

The steps of modeling and deployment have characteristic features that we need to take into account. Cloud platforms make it easier to deal with these features.

Modeling requires **hyperparameter tuning**, finding the parameters that cannot be learned from the data.

![ML Workflow: Hyperparameter Tuning](mlworkflow-modeling-hyperparameter.png)

Deployment requires tracking the model performance; to that end, we need to perform the following tasks:

- Model versioning
- Model monitoring: we track the performance of the model; that way, we can detect drifts and update it.
- Model updating and routing: we deploy new updated models in parallel; we need to be able to route user requests to different models to compare them.

Additionally, note that model predictions can be

- On-demand / online: via API with JSON/XML
    - done all the time, i.e., typical phone web app
    - low latency
    - volume variability, but typically up to 5 MB
- In batch / offline: via files stored on cloud provider (e.g., S3)
    - large volume, done regularly (e.g., weekly)
    - latency is higher, but it's not an issue

![ML Workflow: Hyperparameter Tuning](mlworkflow-deployment.png)

### 1.8 Comparing Cloud Providers

Equivalent / similar systems that cover all 3 steps in the Machine Learning Workflow (explore & process, modeling, deployment):

- [Amazon / AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [Google Vertex AI](https://cloud.google.com/vertex-ai): very similar to SageMaker; maybe SageMaker has more features at the point of the course.
- [Azure AI](https://azure.microsoft.com/en-us/solutions/ai/#platform)

Characteristics of SageMaker:

- It has [built in algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html), e.g.
    - [Linear learner](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)
    - [XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
    - [Factorization machines](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html)
    - [K-means](https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html)
    - [Image classification](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html)
- It also has common frameworks: Scikit-Learn, Pytorch, etc.
- We can use [docker containers](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html) in which we implement our own algorithms
- We can use [Jupyter notebooks](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html)
- We can perform automatic [hyperparameter tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
- We can [monitor models](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-overview.html); we can check the traffic, apply routing, etc.
- We can perform on-demand (online) and batch (offline) predictions; for offline predictions, files need to be stored in S3.

Note that Google doesn't have all these features.

Other systems or cloud providers:

- [Paperspace](https://www.paperspace.com/)
- [Cloud Foundry](https://www.cloudfoundry.org/)

## 2. Building a Model Using AWS SageMaker

SageMaker is basically 2 things:

- Managed **Jupyer notebooks** that run on a virtual machine.
- Access to **AWS APIs** that make all 3 steps of the ML workflow (explore & process, modeling, deployment) much easier. Especially, training and deploying models becomes much easier.

The SageMaker manager notebooks have access to the API. Both the training and the inference tasks performed in the notebooks using the APIs are carried out in a virtual machine. The training task generates model artifacts and the inference task uses those model artifacts, along with the user data.

![SageMaker: Inference Process](./pics/sagemaker_inference.jpg)

### 2.1 Account Setup

- Create a new account to use the [free tier](https://aws.amazon.com/premiumsupport/knowledge-center/what-is-free-tier/)
    - mxagar@gmail.com
- All [Free Tier Offerings](https://aws.amazon.com/free/)
    - 750h EC2: t2.micro or t3.micro
    - 5GB S3
    - 750h RDS (SQL Databases, e.g., PostgreSQL)
    - 750h OpenSearch (log analytics, monitoring, etc.)
    - 1 million API Gateway calls
    - ...
    - 2 Months SageMaker
    - 2 Months RedShift (warehousing)
- After the Free Tier: [Pricing](https://aws.amazon.com/pricing/)

Important: 

- We have an AWS console when we log in.
- All AWS services are accessible from that console: we choose the one we want and create an instance.
- Most important services for machine learning:
    - **S3**: Data Storage
    - **EC2**: Compute, virtual machine instances
    - **SageMaker**: a full solution to perform machine learning; it consists of
        - Notebooks
        - AWS API

#### Regions

Regions are large geographical locations:

- US East
- US West
- Europe
- South America
- ...

Every region has its own independent water and power supply; thus, I understand that system robustness is assured. We need to account for latencies, though. Additionally, **data compliance** issues are handled with regions: EU or US data compliance policies are different.

Availability zones are groups of 1+ data centers within regions, each with independent water and power supply.

Note that:

- We should take the region/zone closest to us, but
- Some services are not available in all regions/zones
- Some services are cheaper in some regions/zones
- They recently opened the EU (Spain) region, but I need to enable it

#### Amazon IAM: Identity and Access Management

Independently of region and tier, we can create user groups and manage their permissions.

By default, we access with our `root account`, which has all permissions, but we should avoid that: instead, we create IAM users immediately and work with them; the `root account` should be used only for minor management tasks.

Imagine that somebody steals your root account and they start mining bitcoin!

### 2.2 Amazon S3: Simple Storage Service

S3 = Simple Storage Service.

We use S3 to store the dataset and the model artifacts.

It is thought for data that is written/modified few times, but read very often.

We have **buckets** or general folders with a global unique name in which we can create sub-folders where files are stored.

Bucket size is unlimited, but each object must be max. of 5 TB.

We can access the data through URIs:

`s3://<bucket_id>.s3.amazonaws.com/<folder>/<file>`

`s3://my-pretty-bucket.s3.amazonaws.com/images/cat.jpg`

`s3://sagemaker-practical-mikel/XGBoost/train/salary.csv`

Also, we can access our buckets via the web interface: Console > S3: Buckets. We can manage the data/buckets there: empty, delete, etc.

#### Storage Tiers

Depending on which tier we use, we can be charged more. In general, tiers are defined according to how often we access the data:

- S3 Standard: frequent access (the most expensive)
- S3 Intelligent-Tiering: varying access
- S3 Standard Infrequent Access (IA): less frequent
- Amazon S3 Glacier: long-term archives, seldom accessed

We assign a tier to a bucket and we can change the tier in time.

#### Creating a Bucket

AWS Dashboard / Management Console > Services > (Storage) S3 > Create Bucket

- Bucket name (unique id): `sagemaker-practical-mikel`
- AWS Region: select one, e.g. `eu-west-3` or `us-east-1`
- ACLs disabled: all object in the bucket owned by the same user
- Block all public access
- Versioning: leave default: disabled
- Encryption: leave default: disabled
- Create bucket!

#### Setting Up the Bucket

After creating it, we click on its id URL: `sagemaker-practical-mikel`

We can upload files, create folders, etc.:

- Create folder: `XGBoost` (encryption: by default, we don't use it)
- We click on folder `XGBoost` and inside of it we select create folder again: `train`
- In `sagemaker-practical-mikel/XGBoost/train`, upload data: `salary.csv`

The URI of the file is 

`s3://sagemaker-practical-mikel/XGBoost/train/salary.csv`

### 2.3 Amazon EC2 Overview: Elastic Compute Cloud

With EC2 we can rent servers in the cloud with different properties and we can easily resize them.

There are several instance types, depending on 

- the compute (CPU, GPU)
- memory capabilities
- and network capabilities we choose.

The more powerful the instance type, the more expensive:

- [Amazon SageMaker Instance Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Available SageMaker Studio Instance Types](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html)

Some generic classification of instance types (SageMaker)

- Standard: `ml.t3.medium`, ...
- Compute optimized: `ml.c5.large`, ...
- Memory optimized: `ml.r5.large`, ...
- Accelerated computing (GPU): `ml.p3.2xlarge`, ...

For instance, in the free tier, currently, we have 250 hours of `ml.t3.medium`.

An example of how to use is given in [Cloud Computing with AWS EC2](#6.-Cloud-Computing-with-AWS-EC2).

In SageMaker, we need to specify the instances we use in our code (usually, in the notebook):

```python
Xgboost_regressor = sagemaker.estimator.Estimator (...,
                                                   train_instance_count=1,
                                                   train_instance_type='ml.t3.medium'
                                                   ...)
```

We have also **inference acceleration** achieved with **Elastic Inference**. With it, we can get realtime inferences. Example: we use train our model and we deploy it to an endpoint t perform inferences. Instead of using a compute instance with GPU, we use one with CPU only, but attach a dedicated GPU to it which accelerates the inference to make it real time.

#### Instance Pricing

See:

- [Amazon SageMaker Instance Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Available SageMaker Studio Instance Types](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html)

**Important**: SageMaker and EC2 instance types are different and their availability depends on the region we choose:

- [SageMaker Instance Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [EC2 Instance Pricing](https://aws.amazon.com/ec2/instance-types/)


Models of pricing (business):

- On-demand: available, pay-per-use, scalable
- Spot instances: we compete with other users, price is much lower (-90%)
- Reserved instances: capacity reserved for 1-3 years (discounts up to 75%)
- Dedicated hosts: physical dedicated server.

### 2.4 Amazon SageMaker: Overview

- Log in to AWS
- AWS Dashboard / Management Console > Services > (Machine Learning) SageMaker.

In SageMaker, everything is organized according to the 3 components of the ML workflow (explore & process, modeling, deployment). Additionally, SageMaker is modular: we can re-use models built in other projects.

In the dashboard, we have many options in the panel of the left:

- Ground Truth: we can create labelling jobs (e.g., with Amazon Turk)
  - When we press **crate labelling job**, we can select the type of data that needs to be labelled: image, text, video
  - Each type has its own properties; e.g., for images: classification, object detection, segmentation, etc.
- Notebook instances (there are templates)
  - Jupyter environment is opened and we can upload / create notebooks
  - There are templates available: SageMake Examples tab
- Training (including hyperparameter tuning)
  - We can select algorithms
  - Select where to dump the artifacts
  - We see all our jobs
- Inference
  - We can create endpoints from which we use the model
- Augmented AI: human reviews
- AWS Marketplace: we can even buy a readily available model!

![AWS SageMaker Dashboard](./pics/sagemaker_dashboard.png)

#### Quotas

AWS assigns quotas or limits to users related to specific services.

We can check the quotas from the AWS Console: Search for Quotas; then search for service SageMaker.

We can request quota increases on the [same quota page](https://us-east-1.console.aws.amazon.com/servicequotas), or in the [AWS support center](https://support.console.aws.amazon.com/support/home?region=us-east-1#/), if desired; usually, we should have the following quotas:

- `ml.m4.xlarge`: 20
- `ml.p2.xlarge`: 1

Note: ALWAYS see in which region we're!

#### AWS SageMaker Studio

AWS SageMaker Studio Overview is a fully integrated IDE. Instead of using the functionalities mentioned before in different instances, we have everything in an IDE similar to R Studio which integrates everything.

Everything can be done in there.

Among others, we can
- Create and work on notebooks
- Create experiments that try different models
- Deploy models
- etc.

Note that 

- The models and all the artifacts product of the training are stored in S3 buckets.
- Additionally, training code is stored in container images, which are collected in the Elastic Container Registry.
- The datasets need to be in Amazon S3 buckets, too.

### 2.5 Set Up a Notebook Instance and Clone the Exercise Repository

We could work with AWS SageMaker Studio; however, we learn here how to instantiate and work with single notebook workspaces.

Left panel > Notebook > Notebook Instances > Create Notebook

- Notebook instance name: `deployment-notebook`
- Notebook instance type: `ml.t2.medium`: we can choose a more powerful one if we'd like to pay
- Elastic inference: we can attach a low cost GPU to accelerate the job; we can leave it `none`.
- Platform identifier: Amazon Linux 2, Jupyter Lab 1 or 3
- Create/choose IAM role:
    - Create one if not done yet.
    - All SageMaker buckets should be accessible.
    - BUT: S3 buckets you specify: None.
    - Give/Enable root access to notebook.
- Git repositories
    - We can clone a repo to the notebook instance.
    - We can do it later, too.
    - The repo we'll work on:
        - https://github.com/mxagar/sagemaker-deployment
- We can ignore the rest.
    - Lifecycle config: no config.
    - Volume size: 5GB
    - Minimum IMDS: 1
- Create the notebook.

**VERY IMPORTANT**:

- In the Notebook Instances list, we wait until the notebook is `InService`, i.e., running. We can use the refresh button. It can take a couple of minutes.
- **ALWAYS Stop** the notebooks that should not run! Otherwise, we are paying!
    - Click on Notebook name hyperlink.
    - Click on 'Stop'. Status changes to 'Stopping'.
    - To restart: Click on 'Start' and wait again.
- **To OPEN** the notebook: Click on 'Open JupyterLab' on the notebook instance list.
- The type of notebook instance we choose determines the compute capacities of it (CPU, RAM, GPU). Usually, it is preferable to use medium/small instances, because the training is performed in other containers. Additionally, note that if we take a small instance, the RAM memory might be limited; thus, it is frequent to remove variables that are persisted to disk, e.g. with `df_train = None`.

#### Notebook

Recall:

- **To OPEN** the notebook: Click on 'Open JupyterLab' on the notebook instance list.
- **ALWAYS Stop** the notebooks that should not run! Otherwise, we are paying!

The notebook starts a virtual machine only for us. We interface with that virtual machine using the Jupyter Lab window!

On the left column menu of the Jupyter Lab window:

- We (can) clone a repo: https://github.com/mxagar/sagemaker-deployment
    - Then it appears in the home of the virtual machine, i.e., we see it in the system directories
- **We can open Amazon SageMaker sample notebooks!**

Like with any Jupyter Lab:

- We can start notebooks with different kernels
- We can start a Terminal! With that terminal we have access to the virtual machine. For instance, we could manually clone a git repo here, inside the folder `~/SageMaker`.

Example of what the Terminal reveals:

```bash
pwd # /home/ec2-user
ll
# anaconda3 
# LICENSE
# Nvidia_Cloud_EULA.pdf
# SageMaker # workspace
# sample-notebooks-1669023763 # examples
# tools
# examples
# nvidia-acknowledgements
# README
# sample-notebooks -> /home/ec2-user/sample-notebooks-1669023763
# src
# tutorials
cd SageMaker
# sagemaker-deployment # the repo cloned via the GUI
# lost+found
```

#### Pushing to HTTPS Repositories

There are several ways to push to repositories that were cloned with their HTTPS version:

1. Use the GIT version instead of HTTPS following the regular method:
    - `ssh-keygen` in the SageMaker notebook instance terminal
    - add the key to `~/.ssh/config`
    - execute `ssh-add`
    - copy the public key to Github (Settings > SSH keys)
2. Using the AWS secret manager: [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-resource.html).
3. Using the Github Personal Access Tokens: **Recommended**.

I used the third method: Personal Access Tokens. Steps:

- Create a token on Github: Github Settings > Developer Settings > Personal Access Token: Create
- In the notebook instance terminal, set user account and activate credential storing:

```bash
# Open Terminal and set user account
git config --global user.email "mxagar@gmail.com"
git config --global user.name "mxagar"

# Activate credential storing to local file
# If we use 
#   credendial.helper cache
# instead of
#   credendial.helper store
# the credential (token) is saved to memory
git config --global credential.helper store

git pull

# Edit something
git add .
git commit -m "message"
git push
# Input
# - username
# - pw: token

# Check that the credential is there!
# If we chose 'store', it should be there
less ~/.git-credentials
```

Later on, to push, either do it in the Terminal, or using the GUI: left menu panel, Git icon.

Note that with the option `credendial.helper store` a file is stored with our credentials, without encryption!

More information:

- [Pushing to HTTPS repositories](https://repost.aws/questions/QU-P1Hlk4OR6K6kAug-wHT_g/can-sagemaker-git-repositories-use-ssh-secrets-no-name-and-password)
- [Git Credentials Storage](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)

### 2.6 Examples: Boston Housing and IMDB Sentiment Analysis

The rest of the module works mainly with 2 examples/projects, which are located in the repository [sagemaker-deployment](https://github.com/mxagar/sagemaker-deployment):

1. The Boston Housing Regression in the folder `Tutorial`, with the following notebooks:

    - `Boston Housing - XGBoost (Batch Transform) - High Level.ipynb`
    - `Boston Housing - XGBoost (Batch Transform) - Low Level.ipynb`
    - `Boston Housing - XGBoost (Deploy) - High Level.ipynb`
    - `Boston Housing - XGBoost (Deploy) - Low Level.ipynb`
    - `Boston Housing - XGBoost (Hyperparameter Tuning) - High Level.ipynb`
    - `Boston Housing - XGBoost (Hyperparameter Tuning) - Low Level.ipynb`
    - `Boston Housing - Updating an Endpoint.ipynb`

2. The IMDB Sentiment Analysis Classification, in the folder `Mini-Projects`, with the following notebooks:

    - `IMDB Sentiment Analysis - XGBoost (Batch Transform) - Solution.ipynb`
    - `IMDB Sentiment Analysis - XGBoost (Batch Transform).ipynb`
    - `IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning) - Solution.ipynb`
    - `IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning).ipynb`
    - `IMDB Sentiment Analysis - XGBoost (Updating a Model) - Solution.ipynb`
    - `IMDB Sentiment Analysis - XGBoost (Updating a Model).ipynb`

The IMDB mini-project is actually a small project that needs to be carried out after following the videos related to the Boston Housing example.

This section focuses with the model building and inference using batch transform; the next sections explain:

- Deployment
- Hyperparamter Tuning
- Updating a Model

### 2.7 Example: Boston Housing: XGBoost Model Batch Transform - High Level

This section starts with the *Boston Housing* example in which a model is built using XGBoost.

Repository: [sagemaker-deployment](https://github.com/mxagar/sagemaker-deployment).

In this section, the notebook `Tutorials / XGBoost (Batch Transform) - High Level.ipynb` is used.

Note that the *high level* label refers to the SageMaker API, which is more high level than the *low level* API. In the section []() the *low level* API is explained; it is interesting to understand what's going on under the hood, which is necessary when we debug the any level API code.

We can select among several kernels; note that the ones with `amazonei` have GPU acceleration; we need to pay for that.

#### Set Up: Session and Role

There are some specific cells in the notebook that are related to SageMaker:

- Session: special object that allows you to do things like manage data in S3 and create and train any machine learning models.
- Role: the IAM role we used for the notebook instance generation.

We need to create these `session` and `role` objects for training. We can create them now or later.

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
session = sagemaker.Session()

# This is an object that represents the IAM role that we are currently assigned. When we construct
# and launch the training job later we will need to tell it what IAM role it should have. Since our
# use case is relatively simple we will simply assign the training job the role we currently have.
role = get_execution_role()
print(role)
```

#### Upload Dataset to S3

SageMaker creates a container for training and inference; to that end, we need to have the training and validation data in S3. We can upload any data from the notebook to S3 using `session`. Note that each notebook instance has an S3 bucket associated to it, and we can connect to it via `session`:

```python
# Load Dataset to local folder on SageMaker notebook VM
boston = load_boston()

# Features & target (median price)
X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# We split the dataset into 2/3 training and 1/3 testing sets.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)

# Then we split the training set further into 2/3 training and 1/3 validation sets.
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)

# Local directory in notebook VM
We need to make sure that it exists.
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# We use pandas to save our test, train and validation data to csv files.
# IMPORTANT: Note that we make sure not to include header
# information or an index as this is required by the built in algorithms provided by Amazon.
X_test.to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

# Also, for the train and
# validation data, it is assumed that the first entry in each row is the target variable.
# BUT NOT for the test dataset split!!
pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

# To save a bit of memory
# we can set the data splits we don't use in the notebook
# explicitly to None.
# However, not ethat these must be saved to file before doing that!
# This step is common, because the VM RAM is usually small,
# so we don't want to run out of memory.
X_train = X_val = Y_val = Y_train = None

###### UPLOAD TO S3

# Each notebook has an S3 bucket.
# The prefix is the folder name in the S3 bucket.
prefix = 'boston-xgboost-HL'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)

```

Once the data has been uploaded to the buckets, we can access it:

- on the web interface: S3,
- via the URIs: see section [2.2 Amazon S3: Simple Storage Service](#2.2-Amazon-S3:-Simple-Storage-Service).

#### Create Training Container and Train Model

In order to train a model, we need to create a docker container for that. With the high-level API, these are the steps:

- We create a training container which has the scripts for training; we need to pass the region we're in and the model/algorithm we'd like to use, i.e., `xgboost`.
- We create an estimator and pass the container to it, as well as additional parameters: instance type, output path, etc.
- We set the [hyperparameters of the estimator](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html).
- We set the dataset (train and validation splits) to the estimator.
- We fit the estimator with the high-level command `fit()`.

In reality, we are creating a **training job** which is executed in a container. We can check information related to that training job in the SageMaker dashboard: Training > Training jobs: Click on the job we want + search 'View logs'. That will open **CloudWatch**.

Checking the logs of the training job with CloudWatch is very important for debugging.

Note that the training jobs we create have a unique name, in the high level API, the name of the container followed by a time stamp; in the low level API the unique name is manually given.

The output of a training job with the high level API is a **model**; a model is in SageMaker

- the training artifact 
- + metadata 
- + information on how to use the artifacts

In the low level API **building the model** means to package all that into a set of files in S3; in the high level API it's done automatically.

We can check our models in the AWS SageMaker web interface: Inference > Models.

Note that models have also unique names.

```python
# As stated above, we use this utility method to construct the image name for the training container.
# We pass: the region we're in and the name of the model/algorithm we'd like to use.
# IMPORTANT: In SageMaker v 2.x, get_image_uri() has been deprecated in favor of
# sagemaker.image_uris.retrieve()
container = get_image_uri(session.boto_region_name, 'xgboost')

# Now that we know which container to use, we can construct the estimator object.
xgb = sagemaker.estimator.Estimator(container, # The image name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

# Hyperparameters:
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
# Note that in our case we have a regression problem, thus: objective='reg:linear'
# In a binary classification, we'd use objective='binary:logistic'
# Look here more possible objective functions & hyperparameters:
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format
s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='csv')

# Train
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

#### Test Trained Model: Batch Transform

We can test the trained model in several ways; one common way is to use the *batch transformer*: an object that takes data in bulk and predicts the target. Note that a container is also started to run the batch transformer; we need to wait for the container to finish and generate the results in a file in S3. After that, we can fetch the output file and load it into the notebook.

In reality, we're creating a **transform job**, which is the container that executes the inference. We can check information about the transform jobs in the AWS SageMaker dashboard: Inference > Batch transform jobs.

```python
# We create the batch transformer object
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
# We transform the test split; we need to pass
# - The S3 location
# - The type of data
# - If the transformer needs to split the data in chunks, where it should do it
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
# Since the transformer runs in a container, we need to wait until it finishes
# before we check the results.
xgb_transformer.wait()

# Fetch results from S3 to notebook VM 
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir

# Load file from notebook VM
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)

# Plot y_true vs y_pred
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

# Clean up to continue using the notebook VM!
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```

### 2.8 Mini-Project: IMDB Sentiment: XGBoost Model Batch Transform - High Level

This section deals with the *IMDB Sentiment Analysis* mini-project in which a model is built using XGBoost.

Repository: [sagemaker-deployment](https://github.com/mxagar/sagemaker-deployment).

In this section, the notebook `Mini-Projects / IMDB Sentiment Analysis - XGBoost (Batch Transform).ipynb` is used.

The mini-project uses the [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), for which an XGBoost model is defined in a very similar fashion as in the previous section.

First, some pre-processing is performed on the dataset:

- Download and unzip it.
- Data is extracted as reviews and labels.
- Data is split into train/test.
- Reviews are processed with the `PorterStemmer()` from NLTK and then converted to data-term matrices with `CountVectorizer()`.

Interestingly, everything is packed in functions which save the steps into pickles and check whether the pickles are already on disk before carrying out any job. that's because the stemming process takes around 1h.

Then, a very similar sequence is followed to train the XGBoost model; nothing new.

### 2.9 Example: Boston Housing: XGBoost Model Batch Transform - Low Level / In Depth

This section shows how to run the *Boston Housing* example using XGBoost but with the **low level** SageMaker API.

Repository: [sagemaker-deployment](https://github.com/mxagar/sagemaker-deployment).

In this section, the notebook `Tutorials / XGBoost (Batch Transform) - Low Level.ipynb` is used.

> The high level approach makes developing new models very straightforward, requiring very little code. The reason this can be done is that certain decisions have been made for you. The low level approach allows you to be far more particular in how you want the various tasks executed, which is good for when you want to do something a little more complicated. Also, when you're debugging the high-level code, we know much better what's going under the hood.

We can select among several kernels; note that the ones with `amazonei` have GPU acceleration; we need to pay for that.

#### Set Up: Session and Role

Same as before.

#### Upload Dataset to S3

Same as before.

#### Create Training Container and Train + Build Model

Now things start to change. We have access to many parameters related to the (1) model and (2) SageMaker.

Additionally,

- We need to create the training job manually; we can check the logs or any metadata of the training job on the AWS SageMaker web interface, as explained in the *low level* implementation: Training > Training jobs.
- We need to **build** the model after it's trained. Building the model in AWS means to package the trained model artifact with metadata and instructions on how to use it. We can check those models on the AWS SageMaker web interface, as explained in the *high level* implementation: Inference > Models. 

```python
# We will need to know the name of the container that we want to use for training. SageMaker provides
# a nice utility method to construct this for us.
container = get_image_uri(session.boto_region_name, 'xgboost')

# We now specify the parameters we wish to use for our training job
training_params = {}

# We need to specify the permissions that this training job will have. For our purposes we can use
# the same permissions that our current SageMaker session has.
training_params['RoleArn'] = role

# Here we describe the algorithm we wish to use. The most important part is the container which
# contains the training code.
training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

# We also need to say where we would like the resulting model artifacts stored.
training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

# We also need to set some parameters for the training job itself. Namely we need to describe what sort of
# compute instance we wish to use along with a stopping condition to handle the case that there is
# some sort of error and the training script doesn't terminate.
training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}
    
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# Next we set the algorithm specific hyperparameters. You may wish to change these to see what effect
# there is on the resulting model.
training_params['HyperParameters'] = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}

# Now we need to tell SageMaker where the data should be retrieved from.
training_params['InputDataConfig'] = [
    {
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": train_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    },
    {
        "ChannelName": "validation",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": val_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    }
]

### Training Job: Definition and Execution

# First we need to choose a training job name. This is useful for if we want to recall information about our
# training job at a later date. Note that SageMaker requires a training job name and that the name needs to
# be unique, which we accomplish by appending the current timestamp.
training_job_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
training_params['TrainingJobName'] = training_job_name

# And now we ask SageMaker to create (and execute) the training job
training_job = session.sagemaker_client.create_training_job(**training_params)

# The training job has been created
# If we want some output about the process & we want to wait until it's done
# we use logs_for_job with wait=True
session.logs_for_job(training_job_name, wait=True)

### Build the Model

# In SageMake, a model is a collection of information
# about a specific algorithm along with the artifacts which result from a training job.
# Building the model means packaging all that information
# We begin by asking SageMaker to describe for us the results of the training job. The data structure
# returned contains a lot more information than we currently need, try checking it out yourself in
# more detail.
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

```

#### Test Trained Model: Batch Transform

In the low level API, again, we can specify many parameters for testing.

We take the example of the batch transformer, i.e., we predict in bulk the outcome of a large dataset.

In the high level API, we created a batch `transformer`; in the low level API we create and configure a transform job. We can check information of those transform jobs on the AWS SageMaker web interface, as explained in the *high level* implementation: Inference > Batch transform jobs.

As before, we need to wait for the container to finish and generate the results in a file in S3. After that, we can fetch the output file and load it into the notebook.

```python
# Just like in each of the previous steps, we need to make sure to name our job and the name should be unique.
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Now we construct the data structure which will describe the batch transform job.
transform_request = \
{
    "TransformJobName": transform_job_name,
    
    # This is the name of the model that we created earlier.
    "ModelName": model_name,
    
    # This describes how many compute instances should be used at once. If you happen to be doing a very large
    # batch transform job it may be worth running multiple compute instances at once.
    "MaxConcurrentTransforms": 1,
    
    # This says how big each individual request sent to the model should be, at most. One of the things that
    # SageMaker does in the background is to split our data up into chunks so that each chunks stays under
    # this size limit.
    "MaxPayloadInMB": 6,
    
    # Sometimes we may want to send only a single sample to our endpoint at a time, however in this case each of
    # the chunks that we send should contain multiple samples of our input data.
    "BatchStrategy": "MultiRecord",
    
    # This next object describes where the output data should be stored. Some of the more advanced options which
    # we don't cover here also describe how SageMaker should collect output from various batches.
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
    },
    
    # Here we describe our input data. Of course, we need to tell SageMaker where on S3 our input data is stored, in
    # addition we need to detail the characteristics of our input data. In particular, since SageMaker may need to
    # split our data up into chunks, it needs to know how the individual samples in our data file appear. In our
    # case each line is its own sample and so we set the split type to 'line'. We also need to tell SageMaker what
    # type of data is being sent, in this case csv, so that it can properly serialize the data.
    "TransformInput": {
        "ContentType": "text/csv",
        "SplitType": "Line",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": test_location,
            }
        }
    },
    
    # And lastly we tell SageMaker what sort of compute instance we would like it to use.
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}

### Execute the transform job

transform_response = session.sagemaker_client.create_transform_job(**transform_request)

# The transform job is being executed in a container
# If we want information on what's going on and know when it's finished
# we need to run wait_for_transform_job
transform_desc = session.wait_for_transform_job(transform_job_name)

### Analyze the results

transform_output = "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)

# Fetch the output from S3 to the VM
!aws s3 cp --recursive $transform_output $data_dir

Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

### clean up

# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir

```

## 3. Deploying and Using a Model

In this section the model is deployed so that it can be used from the outside (production). In SageMaker, deploying a model means to run it in a virtual machine with an endpoint (an URL API); that endpoint is accessible only inside AWS, but later on we add a simple web app which can communicated with the exterior world.

![Model Deployment in Production](./pics/production_environment.jpg)

As explained in the section [2.6 Examples: Boston Housing and IMDB Sentiment Analysis](#2.6-Examples:-Boston-Housing-and-IMDB-Sentiment-Analysis), we have several notebooks for the Boston Housing and IMDB Sentiment Analysis datasets. This section deals with the ones that have the *deploy* and *web app* keywords:

- `Boston Housing - XGBoost (Deploy) - High Level.ipynb`
- `Boston Housing - XGBoost (Deploy) - Low Level.ipynb`
- `IMDB Sentiment Analysis - XGBoost - Web App.ipynb`

### 3.1 Example: Boston Housing: XGBoost Model Deploy - High Level

When we carry out the deployment in SageMaker, instead of running the transform job as before, we `deploy()` the model and send the data to it. The high level API makes that very easy.

So the example notebook is the same as before, but now, after `xgb.fit()`, we run `xgb.deploy()`.

That command creates a virtual machine in which the model is up and we have an **endpoint** waiting for data inputs and ready to send back outputs; that endpoint is basically a URL that is accessible from within AWS for now. The `xgb_predictor` object created by `xgb.deploy()` takes care of all that communication for us! However, the data input/output is serialized.

**VERY IMPORTANT**: We need to stop any deployed model endpoint with `xgb_predictor.delete_endpoint()` when it's not required anymore, because 

Thus, **deployments are endpoints in AWS**; we can check them on the AWS web interface: Inference > Endpoints / Endpoint configurations. Note that *endpoint configurations* only appear in the low level API.


```python
### All as before, until Training...
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})

### Deployment

# When we deploy, we create a virtual machine in which the model is up
# and we have an endpoint waiting for data inputs and ready to send back outputs;
# that endpoint is basically a URL that is accessible from within AWS for now.
# The xgb_predictor object takes care of all that communication for us!
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# We need to tell the endpoint what format the data we are sending is in
# and the data needs to be serialized
#xgb_predictor.content_type = 'text/csv' # this is not necessary anymore
xgb_predictor.serializer = csv_serializer

Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
# predictions is currently a serialized comma delimited string
# and so we would like to break it up as a numpy array.
Y_pred = np.fromstring(Y_pred, sep=',')

# Plot
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

# This is VERY IMPORTANT: the deployed model is running on a virtual machine!
# We need to stop it when not needed, otherwise we need to pay!
# Remember that the costs are proportional to the time up
xgb_predictor.delete_endpoint()

### Clean Up

# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```

### 3.2 Example: Boston Housing: XGBoost Model Deploy - Low Level / In Depth

The notebook in which the low level API is used for the model deployment is very similar to the one with the batch transform, but instead of creating a transform job, we create a deployment endpoint manually.

When the low level API is used, both the *endpoint configuration* and the *endpoint* are created. They can be seen on the AWS SageMaker web interface: Inference > Enpoints / configurations.

```python
### After the model is trained and built

# First, we need to create an endpoint configuration
# As before, we need to give our endpoint configuration a name which should be unique
endpoint_config_name = "boston-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
# The endpoint is like a uniform interface which received/sends data;
# but beneath, we can have several models or ProductionVariants running in parallel.
# We can manually specify the different ProductionVariants/models.
# For now, we deploy a single model.
endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": model_name,
                                "VariantName": "AllTraffic"
                            }])

# After creating an endpoint configuration
# we create an endpoint = deployment of a built model with an URL for IO data transfer.
# Again, we need a unique name for our endpoint
endpoint_name = "boston-xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

### Use the Model: Send data to the Endpoint

# First we need to serialize the input data.
# In this case we want to send the test data as a csv and
# so we manually do this. Of course, there are many other ways to do this.
payload = [[str(entry) for entry in row] for row in X_test.values]
payload = '\n'.join([','.join(row) for row in payload])

# This time we use the sagemaker runtime client rather than the sagemaker client so that we can invoke
# the endpoint that we created.
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = payload)

# We need to make sure that we deserialize the result of our endpoint call.
result = response['Body'].read().decode("utf-8")
Y_pred = np.fromstring(result, sep=',')

# Plot
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

### Shut Down Endpoint!

# We can shut down the endpoint also on the AWS SageMaker web interface:
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)

### Clean Up!

# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir

```

### 3.3 Mini-Project: IMDB Sentiment Analysis: XGBoost Deploy - Web App

This section deals with the notebook

`IMDB Sentiment Analysis - XGBoost - Web App.ipynb`

This notebook uses the high level API and it is very similar to the notebook

`Mini-Projects / IMDB Sentiment Analysis - XGBoost (Batch Transform).ipynb`

These are the differences:

- Dependencies to NLTK and BeautifulSoup are dropped; instead of using them, punctuation & co. are removed with python regex and the `CountVectorizer` performs the tokenization directly. These simplification eases things with Amazon Lambda, used later.
- After we test with a transform job that the model is correctly trained/built, we deploy it with a web app.

So, my comments and code below focus on the web app part.

#### Test the Endpoint

We can test that the endpoint works by sending the test split to it, as done in the batch transform job. The differences are the following:

- The batch transform job creates a container for the job; with the endpoint, we have a container in a VM already running!
- The endpoint cannot handle big chunks of data, so we need to split our test dataset.

Don't forget to stop the endpoint deployment when it's not used!

```python
### Deploy

xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

### Test Endpoint

from sagemaker.predictor import csv_serializer

# We need to tell the endpoint what format the data we are sending is in so that SageMaker can perform the serialization.
#xgb_predictor.content_type = 'text/csv' # variable not existing anymore
xgb_predictor.serializer = csv_serializer

# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])
    
    return np.fromstring(predictions[1:], sep=',')

test_X = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None).values

predictions = predict(test_X)
predictions = [round(num) for num in predictions]

from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)

### Clean Up!

xgb_predictor.delete_endpoint()
```

#### Web App Architecture

This is the web app GUI we'd like for the user:

![Web App GUI](./pics/web_app_gui.jpg)

He/she introduces a review in plain text and the UI should determine whether it's a positive or negative review.

The following figure shows the architecture we're going to use to create that web app:

![Web App Architecture](./pics/web_app_architecture.jpg)

We have the following issues, so far:

- The endpoint of the deployment is accessible to AWS authenticated objects only.
- The user should introduce a plain text, which needs to be processed as a bag or words for the model.

In order to deal with those issues, we're going to generate a new endpoint visible to the world which processes a text string to be a bag of words. To achieve that we use two new AWS technologies/services:

- API Gateway: 
- Lambda functions

**Lambda** is a **function as a service**: until now, we needed to spin up servers/VMs/containers to perform jobs; with lambda we can simply define functions and execute them without spinning up any server. We don't need to have a server running all the time. We tell AWS we want a specific function to be run whenever a specific event occurs. That way, we don't have to pay for a whole server running all the time, but we pay each lambda function call.

**API Gateway** is a service that creates APIs which can connect the outside world with AWS objects. HTTP methods can be used as in a REST API -- see section [1.5 Endpoints and REST APIs](1.5-Endpoints-and-REST-APIs). Thus, we have a public URL to which we can send requests and from which we receive responses. We're going to configure everything so that the API Gateway receives/sends requests/responses and transfers that data to a lambda function. The lambda function will take care of all pre-processing (convert the text into a vector, etc.) and it will communicate with the internal SageMaker endpoint, which is connected to the model.

Anybody can send/receive from the API Gateway. To make things easier, we'll connect a simple HTML web app with a text field that connects to the API Gateway.

Therefore, these are the steps that occur in our we app:

> 1. To begin with, a user will type out a review and enter it into our web app.
> 2. Then, our web app will send that review to an endpoint that we created using API Gateway. This endpoint will be constructed so that anyone (including our web app) can use it.
> 3. API Gateway will forward the data on to the Lambda function
> 4. Once the Lambda function receives the user's review, it will process that review by tokenizing it and then creating a bag of words encoding of the result. After that, it will send the processed review off to our deployed model, i.e., to its original endpoint.
> 5. Once the deployed model performs inference on the processed review, the resulting sentiment will be returned back to the Lambda function.
> 6. Our Lambda function will then return the sentiment result back to our web app using the endpoint that was constructed using API Gateway.

#### Processing a Single Review

We first write the function that, given a plain text, vectorizes the the text according to the vocabulary we have created with `CountVectorizer`

```python
## Proccessing a single review

# Plain text
test_review = "Nothing but a disgusting materialistic pageant of glistening abed remote control greed zombies, totally devoid of any heart or heat. A romantic comedy that has zero romantic chemestry and zero laughs!"

# We use the function we created to clean up: punctuation, HTML
test_words = review_to_words(test_review)
print(test_words)

# We use the vocabulary to create a BoW
# We could use the vectorizer.transform(), too?
def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow

test_bow = bow_encoding(test_words, vocabulary)
print(test_bow)

len(test_bow) # 5000
```

#### Using the SageMaker Endpoint from Outside SageMaker

The deployment endpoints created in a SageMaker session are visible inside SageMaker. In our application we need to connect to them in a Lambda function, which is not in SageMaker. To that end, we can use the `boto3` library, with which we `invoke_endpoint()` given the unique name of the endpoint.

In that invocation, we pass the BoW vector and expect a result from the model. Recall that the input to the endpoint must be a serialized text.

The following code shows

- How to spin up / create a SageMaker endpoint (nothing new).
- How to connect to that endpoint and request a response using `boto3`.

```python
# We create the SageMaker endpoint as always
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

# The (unique) name of the endpoint within SageMaker
# WATCH OUT: this property is deprecated, check the new one
xgb_predictor.endpoint # 'xgboost-2022-11-23-09-27-15-131'

import boto3
# We open a connection to the SageMaker runtime session
runtime = boto3.Session().client('sagemaker-runtime')

# We invoke the endpoint in the SageMaker runtime session we connected to,
# i.e., we send and receive data from that endpoint-model
# Recall that we need to pass the serialized vector as a text/string
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint, # The name of the endpoint we created
                                       ContentType = 'text/csv',                     # The data format that is expected
                                       Body = ','.join([str(val) for val in test_bow]).encode('utf-8'))

# The response is an HTML object
# We need the Body object within it
print(response)
# {'ResponseMetadata': ... 'Body': <botocore.response.StreamingBody object at 0x7f59aae6b4c0>}

# We extract the Body part of the response
# THAT'S THE SENTIMENT SCORE!
response = response['Body'].read().decode('utf-8')
print(response) # 0.37377044558525085

# Terminate SageMaker endpoint if not used
xgb_predictor.delete_endpoint()
```

#### Building an AWS Lambda Function

Our goal is to create and connect the following elements:

- An HTML web app which reads a text field and connects to a REST API
- An AWS API Gateway which works as a public REST API
- An AWS Lambda function which is connected to the AWS API Gateway: it receives the text, processes it and sends it to the SageMaker deployment endpoint, which is reachable only within AWS.
- An AWS SageMaker deployment endpoint, which contains the model on a VM and communicates with the lambda function.

In this section, we'll focus on the Lambda function. Recall that lambda functions act like *serverless* functions that are executed on an event. Our event will be an input from the AWS Gateway.

Another typical use of Lambda: When data is uploaded to an S3 bucket, process that data and insert it into a database.

Basic requirements of Lambda functions:

- The amount of code contained in a Lambda should be relatively small.
- We need to have an IAM role for Lambda which has full access to SageMaker.

Thus, **first, we need to create an IAM role**:

- AWS Console, Search IAM = Identity and Access Management
- IAM Dashboard: Roles
- Create role
- Select: AWS Service, Lambda; Next; (our role is going to be for Lambda)
- Search 'SageMaker' and select the policy 'AmazonSageMakerFullAccess'; Next
- Role name: LambdaSageMakerRole; we can choose another one, but this is quite descriptive
- Create role

**Second, we create the Lambda function**:

- AWS Console, Search Lambda
- Lambda Dashborad: Create a Lambda function
- Author from scratch
- Function name: sentiment_lambda_function
- Runtime: Python 3.X
- Under Permissions: Change default existing role
- Use existing role: LambdaSageMakerRole
- Create function

If we scroll down on the AWS Lambda function page, we'll see the box where the lambda function code needs to be added. The code is below.

Note that the structure of a Lambda function code is the following:

- All imports necessary.
- The definition of all functions and variables used.
- `lambda_handler()`: the lambda function executed; it has an `event` object which contains the information used inside the function -- in our case, that's the plain text from the API Gateway.

In the particular case of our web app, we need to get two values from the SageMaker notebook:

- The vocabulary dictionary: `print(str(vocabulary))`.
- The name of the SageMaker endpoint to which we'd like to connect: `xgb_predictor.endpoint`

```python
# We need to use the low-level library to interact with SageMaker since the SageMaker API
# is not available natively through Lambda.
import boto3

# And we need the regular expression library to do some of the data processing
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow


def lambda_handler(event, context):

    # In our SageMaker notebook, we print the vocabulary
    # print(str(vocabulary))
    # and copy all its printed value here!
    # {'was': 4805, 'really': 3556, ...}
    vocab = "*** ACTUAL VOCABULARY GOES HERE ***"

    # The event object is accessed here: it contains the plain text!
    words = review_to_words(event['body'])
    bow = bow_encoding(words, vocab)

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    # In our SageMaker notebook, we print the name of the endpoint
    # xgb_predictor.endpoint: 'xgboost-2022-11-23-12-04-57-086'
    # and copy all its printed value here!
    # Don't forget to serialize the vectorized text!
    response = runtime.invoke_endpoint(EndpointName = '***ENDPOINT NAME HERE***',# The name of the endpoint we created
                                       ContentType = 'text/csv',                 # The data format that is expected
                                       Body = ','.join([str(val) for val in bow]).encode('utf-8')) # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    # Round the result so that our web app only gets '1' or '0' as a response.
    result = round(float(result))

    # We need to return a result, which is an HTTP response with 3 elements:
    # - Status Code: if data successfully received, code should start with 2, e.g., 200.
    # - HTTP Headers: data format in the message, additional info, etc.
    # - Message: the output data sent to the user, i.e., the prediction.
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }
```

**Finally, we can perform **Lambda Tests** on the AWS Lambda web interface:

- Above the code window, on the Test button, select drop-down arrow and click on 'Configure test events'
- Create new event
- Template: API Gateway AWS Proxy
- Event name: testEvent
- We get a event JSON which will be sent to our Lambda function for testing purposes. Note that this object is passed as `event` to the `lambda_handler()`; and the element from that object which is used is `event['body']`, i.e., the review plain text.
- We add a test review to the `'body'` key from the event JSON, e.g., "This movie was terrible. Nobody should watch it!"
- Save/create test event
- Deploy
- Test

Now, we should get a response like this:

```json
Response
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "text/plain",
    "Access-Control-Allow-Origin": "*"
  },
  "body": "0"
}
```

The Lambda function is deployed and the test works: we get a response in "body" of 0, i.e., negative sentiment to the bad test review text we wrote!

#### Creating an AWS API Gateway

The AWS API Gateway is necessary to connect our Lambda function and the internal SageMaker endpoint hanging from it to the world.

- AWS Console, Search API Gateway
- Create/Build, REST API (not private)
- Select: REST, New API
- API name: sentimentAnalysis
- Endpoint type: Regional (default)
- Create API

Now, we can **define the API**:

- Actions, Create method: POST; select checkmark
- Integration type: Lambda
- Select: Use Lambda Proxy Integration; that means the API Gateway is not going to do any data checking/processing, i.e., the API Gateway will only send the data to the lambda and receive and return its response.
- Lambda function: sentiment_lambda_function (start typing and TAB; the name we entered before appears).
- Save; accept we want to give permissions.

Now, we have defined the API and we need to **deploy the API**:

- Actions, Deploy API
- Deployment stage: Create new, call it prod, for production; a deployment stage is like a deployment environment: production, test, etc.
- Save changes

That's it. We get an invoke URL, e.g.,

`https://55yo1tq9p2.execute-api.us-east-1.amazonaws.com/prod`

If we send a POST request with a plan text review, we'll get as a response the sentiment score!

#### Web App HTML File

In the folder of the SageMaker notebook we have a web GUI template with a text field: `index.html`.

We need to add our API Gateway URL to it, download the `index.html`, open it with the browser locally, and there we have our web app!

```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Sentiment Analysis Web App</title>
        <meta charset="utf-8">
        <meta name="viewport"  content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>

        <script>
         "use strict";
         function submitForm(oFormElement) {
             var xhr = new XMLHttpRequest();
             xhr.onload = function() {
                 var result = parseFloat(xhr.responseText);
                 var resultElement = document.getElementById('result');
                 if (result == 0) {
                     resultElement.className = 'bg-danger';
                     resultElement.innerHTML = 'Your review was NEGATIVE!';
                 } else {
                     resultElement.className = 'bg-success';
                     resultElement.innerHTML = 'Your review was POSITIVE!';
                 }
             }
             xhr.open (oFormElement.method, oFormElement.action, true);
             var review = document.getElementById('review');
             xhr.send (review.value);
             return false;
         }
        </script>

    </head>
    <body>

        <div class="container">
            <h1>Is your review positive, or negative?</h1>
            <p>Enter your review below and click submit to find out...</p>
            <form method="POST"
                  action="https://55yo1tq9p2.execute-api.us-east-1.amazonaws.com/prod"
                  onsubmit="return submitForm(this);" >                     <!-- HERE IS WHERE YOU NEED TO ENTER THE API URL -->
                <div class="form-group">
                    <label for="review">Review:</label>
                    <textarea class="form-control"  rows="5" id="review">Please write your review here.</textarea>
                </div>
                <button type="submit" class="btn btn-default">Submit</button>
            </form>
            <h1 class="bg-success" id="result"></h1>
        </div>
    </body>
</html>

```

#### Shut Everything Down

- **Very important**: terminate the SageMaker endpoint, via the web interface or with `xgb_predictor.delete_endpoint()` in the SageMaker notebook.
- Clean up the data in the SageMaker notebook instance VM.
- Remove the API Gateway (via the web interface: select the API, Actions, Delete). We could leave it, because the cost is per use, but just in case.
- Remove the Lambda (via the web interface: select the function, Actions, Delete). We could leave it, because the cost is per use, but just in case.

## 4. Hyperparameter Tuning

Hyperparameters are parameters that cannot be learned by the model, instead they define the model and its learning. Instead of fixing their value, we want to define ranges of values and let SageMaker find the optimum set. That's hyperparameter tuning.

SageMaker has a `HyperparameterTuner`, similar to `GridSearchCV` in `sklearn`; however, by default no grid search is done, but Bayesian optimization is applied. This approach treats hyperparameter tuning as a regression problem in which the metric needs to be optimized based on the hyperparameter values. Training jobs are performed one after the other using the knowledge from previous trainings and explore/exploit strategies. The outcome delivers a training job with the probably best set of hyperparameters.

See: [How Hyperparameter Tuning Works](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html).

This section deals with these Boston Housing notebooks from the repository:

- `Tutorials / Boston Housing - XGBoost (Hyperparameter Tuning) - High Level.ipynb`
- `Tutorials / Boston Housing - XGBoost (Hyperparameter Tuning) - Low Level.ipynb`

The first explains how the *high level* API works, whereas the second deeps dive into the *low level* API.

Additionally, the following IMBD mini-project notebook completed:

`Mini-Projects / IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning).ipynb`

### 4.1 Boston Hyperparameter Tuning: High Level API

To perform hyperparameter tuning with the high level API, we follow these steps:

- We create a base estimator with base hyperparameters.
- We define a `HyperparameterTuner` with parameter ranges and total number of sets to test.
- We get the name of the best training job according to the metric we've defined.
- We **attach** the best training job to an empty estimator.

Attaching a job to an estimator means in practice taking an existing model we've trained. So it's not exclusive to hyperparameter tuning, i.e., we can look at the training jobs we have and pick one we'd like as an estimator!

```python
### Prepare and Upload the Data
### ... as always

### Train the Model / Hyperparameter Tuning

# As stated above, we use this utility method to construct the image name for the training container.
container = get_image_uri(session.boto_region_name, 'xgboost')

# Now that we know which container to use, we can construct the estimator object.
# The estimator object is defined as always, but it will be used for hyperparameter tuning,
# i.e., many models are going to be branched from it
xgb = sagemaker.estimator.Estimator(container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

# We need to set default values for the hyperparameters, i.e., we define the base model
# which will be modified varying the hyperparameters
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
# The HyperparameterTuner defines how to change the base model
# and how to compare them.
# We need to define the metric & split to which it is applied,
# as well as the maximum number of models we'd like to train and test,
# and the ranges of the hyperparameter values.
# Metrics for XGBoost:
# - regression: validation:rmse, validation:mse
# - binary classification: validation:error, validation:f1, ...
# - more: https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
xgb_hyperparameter_tuner = HyperparameterTuner(estimator = xgb, # The estimator object to use as the basis for the training jobs.
                                               objective_metric_name = 'validation:rmse', # The metric used to compare trained models.
                                               objective_type = 'Minimize', # Whether we wish to minimize or maximize the metric.
                                               max_jobs = 20, # The total number of models to train in total
                                               max_parallel_jobs = 3, # The number of models to train in parallel
                                               hyperparameter_ranges = {
                                                    'max_depth': IntegerParameter(3, 12),
                                                    'eta'      : ContinuousParameter(0.05, 0.5),
                                                    'min_child_weight': IntegerParameter(2, 8),
                                                    'subsample': ContinuousParameter(0.5, 0.9),
                                                    'gamma': ContinuousParameter(0, 10),
                                               })

# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format.
s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='csv')

# We simply call fit() on the HyperparameterTuner
xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})

xgb_hyperparameter_tuner.wait()

# We get the best training job: the model with the best hyperparameters
# BUT this is only the name of the job
xgb_hyperparameter_tuner.best_training_job() # 'xgboost-221124-0649-020-157431cc'

# Still, we need to construct the best estimator
# We can use attach() for that: an estimator is created and the best training job
# is attached to it.
# Attach can be used in other cases too:
# If we have a training job name with the performance we want (e.g., a past traine model)
# we can simple create an estimator and attach the training job (name) to it!
xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())

### Test the Model
# As always!

# Now, we can use the batch transformer to test the best estimator
# as always
xgb_transformer = xgb_attached.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')

xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')

xgb_transformer.wait()

# Fetch the results
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir

Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

### Clean Up!

# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```

### 4.2 IMBD Hyperparameter Tuning: High Level API

This section deals with the notebook

`IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning).ipynb`.

It is a mini-project/exercise in which the former section needs to be re-implemented.

### 4.3 Boston Hyperparameter Tuning: Low Level API


## 5. Updating a Model

## 6. Cloud Computing with AWS EC2

We need to perform two tasks:

1. Launch an EC2 instance
2. Connect from our computer to that EC2 instance

Note: I copied this section from my other notes in [`CVND_CloudComputing.md`](https://github.com/mxagar/computer_vision_udacity/blob/main/02_Cloud_Computing/CVND_CloudComputing.md).

### 6.1 Launch EC2 Instances

EC2 = Elastic Compute Cloud. We can launch VM instances.

Create an AWS account, log in to the AWS console & search for "EC2" in the services.

Select region on menu, top-right: Ireland, `eu-west-1`. Selecting a region **very important**, since everything is server region specific. Take into account that won't see the instances you have in different regions than the one you select in the menu! Additionally, we should select the region which is closest to us. Not also that not all regions have the same services and the service prices vary between regions!

Press: **Launch Instance**.

Follow these steps:

1. Choose an Amazon Machine Image (AMI) - An AMI is a template that contains the software configuration (operating system, application server, and applications) required to launch your instance. I looked for specific AMIs on the search bar (keyword "deep learning") and selected `Deep Learning AMI (Amazon Linux 2) Version 61.3` and `Deep Learning AMI (Amazon Linux 2) Version 61.3` for different instances. Depending on which we use, we need to install different dependencies.

2. Choose an Instance Type - Instance Type offers varying combinations of CPUs, memory (GB), storage (GB), types of network performance, and availability of IPv6 support. AWS offers a variety of Instance Types, broadly categorized in 5 categories. You can choose an Instance Type that fits our use case. The specific type of GPU instance you should launch for this tutorial is called `p2.xlarge` (P2 family). I asked to increase the limit for EC2 in the support/EC2-Limits menu option to select `p2.xlarge`, but they did not grant it to me; meanwhile, I chose `t2.micro`, elegible for the free tier.

3. Configure Instance Details - Provide the instance count and configuration details, such as, network, subnet, behavior, monitoring, etc.

4. Add Storage - You can choose to attach either SSD or Standard Magnetic drive to your instance. Each instance type has its own minimum storage requirement.

5. Add Tags - A tag serves as a label that you can attach to multiple AWS resources, such as volumes, instances or both.

6. Configure Security Group - Attach a set of firewall rules to your instance(s) that controls the incoming traffic to your instance(s). You can select or create a new security group; when you create one:
    - Select: Allow SSH traffic from anywhere
    - Then, when you launch the instance, **you edit the security group later**
    - We can also select an existing security group

7. Review - Review your instance launch details before the launch.

8. I was asked to create a key-pair; I created one with the name `face-keypoints` using RSA. You can use a key pair to securely connect to your instance. Ensure that you have access to the selected key pair before you launch the instance. A file `face-keypoints.pem` was automatically downloaded.

More on [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/)

Important: Editing the security group: left menu, `Network & Security` > `Security Groups`:

- Select the security group associated with the created instance (look in EC2 dashboard table)
- Inbound rules (manage/create/add rule):
    - SSH, 0.0.0.0/0, Port 22
    - Jupyter, 0.0.0.0/0, Port 8888
    - HTTPS (Github), 0.0.0.0/0, Port 443
- Outbound rules (manage/create/add rule):
    - SSH, 0.0.0.0/0, Port 22
    - Jupyter, 0.0.0.0/0, Port 8888
    - HTTPS (Github), 0.0.0.0/0, Port 443

If we don't edit the security group, we won't be able to communicate with the instance in the required ports!

**Important: Always shut down / stop all instances if not in use to avoid costs! We can re-start afterwards!**. AWS charges primarily for running instances, so most of the charges will cease once you stop the instance. However, there are smaller storage charges that continue to accrue until you **terminate** (i.e. delete) the instance.

We can also set billing alarms.

### 6.2 Connect to an Instance

Once the instance is created, 

1. We `start` it: 

    - EC2 dashboard
    - Instances
    - Select instance
    - Instance state > Start

2. We connect to it from our local shell

```bash
# Go to the folder where the instance key pem file is located
cd .../project
# Make sure the pem file is only readable by me
chmod 400 face-keypoints.pem
# Connect to instance
# user: 'ec2-user' if Amazon Image, 'ubuntu' if Ubuntu image
# Public IP: DNS or IP number specified in AWS EC2 instance properties
# ssh -i <pem-filename>.pem <user>@<public-IP>
ssh -i face-keypoints.pem ec2-user@3.248.188.159
# We need to generate a jupyter config file
jupyter notebook --generate-config
# Make sure that
# ~/.jupyter/jupyter_notebook_config.py
# contains 
# c.NotebookApp.ip = '*'
# Or, alternatively, directly change it:
sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip = '*'/g" ~/.jupyter/jupyter_notebook_config.py
# Clone or download the code
# Note that the SSH version of the repo URL cannot be downloaded;
# I understand that's because the SSH version is user-bound 
git clone https://github.com/mxagar/P1_Facial_Keypoints.git
# Go to downloaded repo
cd P1_Facial_Keypoints
# When I tried to install the repo dependencies
# I got some version errors, so I stopped and
# I did not install the dependencies.
# However, in a regular situation, we would need to install them.
# Also, maybe:
# pip install --upgrade setuptools.
sudo python3 -m pip install -r requirements.txt
# Launch the Jupyter notebook without a browser
jupyter notebook --ip=0.0.0.0 --no-browser
# IMPORTANT: catch/copy the token string value displayed:
# http://127.0.0.1:8888/?token=<token-string>
```

3. Open our local browser on this URL, composed by the public IP of the EC2 instance we have running and the Jupyter token:

```
http://<public-IP>:8888/?token=<token-string>
```

### 6.3 Pricing

Always stop & terminate instances that we don't need! Terminate erases any data we have on the instance!

[Amazon EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)

## 7. Google Colab

See [`Google_Colab_Notes.md`](https://github.com/mxagar/computer_vision_udacity/blob/main/02_Cloud_Computing/Google_Colab_Notes.md).

## 8. Project: Deploying a Sentiment Analysis Model

See repository: []().

