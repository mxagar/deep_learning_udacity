'''This file contains a summary of the most important
steps necessary when working with machine learning models
in AWS SageMaker:

1. Setup
	- Imports
	- Create session and role objects
2. Load dataset
3. Pre-process dataset
	- Basic Cleaning and Feature Engineering
4. Upload pre-processed dataset to S3
	- Split dataset and save to file in VM
	- Upload saved files to S3
5. Training: High-Level API
	- Select algorithm by creating a container of its type
	- Set hyperparameters
	- Train: Training job info on AWS console
6. Batch Transform: High-Level API
	- Create the batch transformer object and run it
	- Fetch results from S3 to notebook VM and evaluate
	- Clean up
7. Training: Low-Level API
	- Define training parameters
	- Training job: Definition and execution
	- Build the model = Package artifacts in usable model file
8. Batch Transform: Low-Level API
	- Define batch transform job
	- Execute the transform job
	- Fetch and analyze the results
	- Clean up
9. Deploy model endpoint: High-Level API
	- Deploy endpoint
	- Predict, de-serialize, evaluate
	- Shut down endpoint
	- Clean up
10. Deploy model endpoint: Low-Level API
	- Create endpoint configuration: visible in AWS console
	- Deploy endpoint
	- Use the model: Send data to the endpoint and receive a result
	- Shut down endpoint!
	- Clean up
11. Hyperparameter Tuning: High-Level API
	- Select algorithm by creating a container of its type
	- Set base hyperparameters
	- Define hyperparameter tuner
	- Train models with varied hyperparameters
	- Create best estimator by attaching the best training job
	- Test the model (e.g., with a batch transform job)
	- Clean up
12. Hyperparameter Tuning: Low-Level API
	- Define base training parameters
	- Define and run tuning job
	- Build the model: pick best training job and package artifact to be a valid model
	- Test the model (e.g., with a batch transform job)
	- Clean up
13. Define Combined Endpoints and Update Endpoints/Models
	- Model 1: Container definition, training, building
	- Model 2: Container definition, training, building
	- First endpoint configuration and deployment: Two models combined
	- Use first endpoint
	- Second endpoint configuration: One model
	- Update first endpoint with configuration of second
	- Shut down endpoint
	- Clean up
14. Using the Endpoint from Outside: Lambda + API Gateway + Web App
	- Create IAM role for Lambda
	- Create a Lambda function
		- Definitions: vocabulary
		- Take text in event['body'] from API Gateway and pre-process it: review_to_words()
		- Create a Bag of Words (BOW) of the text using the vocabulary
		- Invoke SageMaker endpoint and pass serialized BoW
		- Catch result from SageMaker endpoint and pack as response for API Gateway
	- Create an API Gateway (REST) with POST method and Lambda proxy integration
	- HTML web app with interaction to API Gateway URL

IMPORTANT NOTES:

- This file does not run; it is a collection of API calls that can be re-used.
- Minimum explanations are given here.
- Check the co-located picture: sagemaker_examples_workflow.png

For a basic understanding of cloud technologies,
focusing on AWS SageMaker, as well the explanation of some calls & parameters,
check the co-located file DLND_Deployment.md; Github link:

https://github.com/mxagar/deep_learning_udacity/blob/main/06_Deployment/DLND_Deployment.md

Author: Mikel Sagardia
Date: 2022.11.29
'''

########################
### 1. Setup
########################

## Imports

import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

## Create session and role objects

session = sagemaker.Session()
role = get_execution_role()


########################
### 2. Load dataset
########################

boston = load_boston()

X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)


########################
### 3. Pre-process dataset
########################

## Basic Cleaning and Feature Engineering - We can do anything here.


########################
### 4. Upload pre-processed dataset to S3
########################

## Split dataset and save to file in VM

# Dataset -> Train / Test
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
	X_bos_pd,
	Y_bos_pd,
	test_size=0.33)

# Train -> Train / Validation
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(
	X_train,
	Y_train,
	test_size=0.33)

# Check directory is there, else create it
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Test: remove header + index
X_test.to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

# Train & Validation: remove header + index and put target as first column
pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

# Save memory!
X_train = X_val = Y_val = Y_train = None

## Upload saved files to S3

# Each notebook has an S3 bucket; prefix is folder name in bucket
prefix = 'boston-xgboost-HL'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)

########################
### 5. Training: High-Level API
########################

## Select algorithm by creating a container of its type

# Other algos: 'linear-learner', etc.
container = get_image_uri(session.boto_region_name, 'xgboost')

# Estimator object: pass algorithm container
xgb = sagemaker.estimator.Estimator(container, # The image name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

## Set hyperparameters

# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
# Regression: objective='reg:linear'
# Binary classification: objective='binary:logistic'
# More possible objective functions & hyperparameters:
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst#learning-task-parameters
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='csv')

## Train: Training job info on AWS console

# Watch out: no model us built yet!
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})

########################
### 6. Batch Transform: High-Level API
########################

## Create the batch transformer object and run it

xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')

# Transform: data location, type of data, where to cut if necessary
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
xgb_transformer.wait()

## Fetch results from S3 to notebook VM and evaluate

!aws s3 cp --recursive $xgb_transformer.output_path $data_dir

# Load file from notebook VM
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)

# Plot y_true vs y_pred
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Clean up

!rm $data_dir/*
!rmdir $data_dir

########################
### 7. Training: Low-Level API
########################

# WARNING:
# This section continues from the end of section '4. Upload pre-processed dataset to S3'

# Select algorithm by creating a container of its type
# Other algos: 'linear-learner', etc.
container = get_image_uri(session.boto_region_name, 'xgboost')

## Define training parameters

training_params = {}

training_params['RoleArn'] = role

training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}
    
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# Hyperparameters
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

## Training job: Definition and execution

# Training job name: must be unique, thus append timestamp
training_job_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
training_params['TrainingJobName'] = training_job_name

training_job = session.sagemaker_client.create_training_job(**training_params)

# Train
session.logs_for_job(training_job_name, wait=True)

## Build the model = Package artifacts in usable model file

training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

########################
### 8. Batch Transform: Low-Level API
########################

## Define batch transform job

# Transform job name: must be unique, thus, use timestamp
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

transform_request = \
{
    "TransformJobName": transform_job_name,
    
    "ModelName": model_name,
    
    "MaxConcurrentTransforms": 1,
    
    "MaxPayloadInMB": 6,
    
    "BatchStrategy": "MultiRecord",
    
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
    },
    
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
    
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}

## Execute the transform job

transform_response = session.sagemaker_client.create_transform_job(**transform_request)
transform_desc = session.wait_for_transform_job(transform_job_name)

## Fetch and analyze the results

transform_output = "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)

!aws s3 cp --recursive $transform_output $data_dir

Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Clean up

!rm $data_dir/*
!rmdir $data_dir


########################
### 9. Deploy model endpoint: High-Level API
########################

# WARNING:
# This section continues from the end of section '5. Training: High-Level API'

## Deploy endpoint

# We have performed xgb.git() before this
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

xgb_predictor.serializer = csv_serializer

## Predict, de-serialize, evaluate

# Predictions is currently a serialized comma delimited string
# and so we would like to break it up as a numpy array.
Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
Y_pred = np.fromstring(Y_pred, sep=',')

# Plot
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Shut down endpoint

# VERY IMPORTANT: the deployed model is running on a virtual machine!
# We need to stop it when not needed, otherwise we need to pay!
xgb_predictor.delete_endpoint()

## Clean up

!rm $data_dir/*
!rmdir $data_dir


########################
### 10. Deploy model endpoint: Low-Level API
########################

# WARNING:
# This section continues from the end of section '7. Training: Low-Level API'

## Create endpoint configuration: visible in AWS console

# Unique endpoint configuration name: use timestamp
endpoint_config_name = "boston-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": model_name,
                                "VariantName": "AllTraffic"
                            }])

## Deploy endpoint

# Unique endpoint name: use timestamp
endpoint_name = "boston-xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

## Use the model: Send data to the endpoint and receive a result

# Serialize the input data
payload = [[str(entry) for entry in row] for row in X_test.values]
payload = '\n'.join([','.join(row) for row in payload])

# Invoke the endpoint
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = payload)

# De-serialize response
result = response['Body'].read().decode("utf-8")
Y_pred = np.fromstring(result, sep=',')

# Plot
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Shut down endpoint!

# Also possible in the AWS console
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)

## Clean up

!rm $data_dir/*
!rmdir $data_dir


########################
### 11. Hyperparameter Tuning: High-Level API
########################

# WARNING:
# This section continues from the end of section '4. Upload pre-processed dataset to S3'

## Select algorithm by creating a container of its type

# Other algos: 'linear-learner', etc.
container = get_image_uri(session.boto_region_name, 'xgboost')

# Estimator object: pass algorithm container
xgb = sagemaker.estimator.Estimator(container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

## Set base hyperparameters

# Base hyperparameters: these will be varied
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

## Define hyperparameter tuner

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

s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='csv')

## Train models with varied hyperparameters

xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
xgb_hyperparameter_tuner.wait()

# Get name of best job
xgb_hyperparameter_tuner.best_training_job() # 'xgboost-221124-0649-020-157431cc'

## Create best estimator by attaching the best training job

# An estimator is created and the training job we decide is is attached to it
xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())

## Test the model (e.g., with a batch transform job)

xgb_transformer = xgb_attached.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
xgb_transformer.wait()

# Fetch the results and evaluate
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir

Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Clean up

!rm $data_dir/*
!rmdir $data_dir


########################
### 12. Hyperparameter Tuning: Low-Level API
########################

# WARNING:
# This section continues from the end of section '4. Upload pre-processed dataset to S3'

# Select algorithm by creating a container of its type
# Other algos: 'linear-learner', etc.
container = get_image_uri(session.boto_region_name, 'xgboost')

## Define base training parameters

training_params = {}

training_params['RoleArn'] = role

training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}
    
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# We are setting up
# a training job which will serve as the base training job
# for the eventual hyperparameter
# tuning job, we only specify the _static_ hyperparameters.
# That is, the hyperparameters that
# we do _not_ want SageMaker to change.
training_params['StaticHyperParameters'] = {
    "gamma": "4",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}

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

## Define and run tuning job

# In the tuning job configuration
# we define how the base training job needs to be derived
# with varied hyperparameters.
# The ranges and hyperparameter names are defined
# as well as the tuning strategy, e.g., Bayesian optimization.

tuning_job_config = {
    # First we specify which hyperparameters we want SageMaker to be able to vary,
    # and we specify the type and range of the hyperparameters.
    "ParameterRanges": {
    "CategoricalParameterRanges": [],
    "ContinuousParameterRanges": [
        {
            "MaxValue": "0.5",
            "MinValue": "0.05",
            "Name": "eta"
        },
    ],
    "IntegerParameterRanges": [
        {
            "MaxValue": "12",
            "MinValue": "3",
            "Name": "max_depth"
        },
        {
            "MaxValue": "8",
            "MinValue": "2",
            "Name": "min_child_weight"
        }
    ]},
    # We also need to specify how many models should be fit and how many can be fit in parallel
    "ResourceLimits": {
        "MaxNumberOfTrainingJobs": 20,
        "MaxParallelTrainingJobs": 3
    },
    # Here we specify how SageMaker should update the hyperparameters as new models are fit
    "Strategy": "Bayesian",
    # And lastly we need to specify how we'd like to determine which models are better or worse
    "HyperParameterTuningJobObjective": {
        "MetricName": "validation:rmse",
        "Type": "Minimize"
    }
  }

# Tuning job name: must be unique, thus, use timestamp (but, limited to 32 chars)
tuning_job_name = "tuning-job" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

session.sagemaker_client.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                                           HyperParameterTuningJobConfig = tuning_job_config,
                                                           TrainingJobDefinition = training_params)

session.wait_for_tuning_job(tuning_job_name)

## Build the model: pick best training job and package artifact to be a valid model

tuning_job_info = session.sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

best_training_job_name = tuning_job_info['BestTrainingJob']['TrainingJobName']
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=best_training_job_name)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

# Unique name
model_name = best_training_job_name + "-model"

# We define which container/algorithm we'd like to use
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)

## Test the model (e.g., with a batch transform job)

# Create a transform job
# Unique name
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

transform_request = \
{
    "TransformJobName": transform_job_name,
    
    "ModelName": model_name,
    
    "MaxConcurrentTransforms": 1,
    
    "MaxPayloadInMB": 6,
    
    "BatchStrategy": "MultiRecord",
    
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
    },
    
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
    
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}

transform_response = session.sagemaker_client.create_transform_job(**transform_request)

transform_desc = session.wait_for_transform_job(transform_job_name)

# Fetch the results
transform_output = "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)

!aws s3 cp --recursive $transform_output $data_dir

# Evaluate
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Clean up

!rm $data_dir/*
!rmdir $data_dir

########################
### 13. Define Combined Endpoints and Update Endpoints/Models
########################

# WARNING:
# This section continues from the end of section '4. Upload pre-processed dataset to S3'
# In this section two models are trained with the high-level API:
# 1. an XGBoost model
# 2. and a Linear Learner
# Then, using the low-level API:
# - The models are built
# - One endpoint configuration which combines both models is defined and deployed
# - Another endpoint configuration which has a unique model is defined
# - The first endpoint is updated to run the second configuration
# IMPORTANT: For the user, the endpoint interface (URL)
# remains to be the same, but internally it's changed,
# because another set of models is started.

## Model 1: Container definition, training, building

# Define
xgb_container = get_image_uri(session.boto_region_name, 'xgboost')
xgb = sagemaker.estimator.Estimator(xgb_container, # The image name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

# Train
s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='csv')
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})

# Build
xgb_model_name = "boston-update-xgboost-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
xgb_primary_container = {
    "Image": xgb_container,
    "ModelDataUrl": xgb.model_data
}
xgb_model_info = session.sagemaker_client.create_model(
                                ModelName = xgb_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = xgb_primary_container)

## Model 2: Container definition, training, building

# Define
linear_container = get_image_uri(session.boto_region_name, 'linear-learner')
linear = sagemaker.estimator.Estimator(linear_container, # The name of the training container
                                        role,      # The IAM role to use (our current role in this case)
                                        train_instance_count=1, # The number of instances to use for training
                                        train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                        output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                            # Where to save the output (the model artifacts)
                                        sagemaker_session=session) # The current SageMaker session
linear.set_hyperparameters(feature_dim=13, # Our data has 13 feature columns
                           predictor_type='regressor', # We wish to create a regression model
                           mini_batch_size=200) # Here we set how many samples to look at in each iteration

# Train
s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='text/csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='text/csv')
linear.fit({'train': s3_input_train, 'validation': s3_input_validation})

# Build
linear_model_name = "boston-update-linear-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
linear_primary_container = {
    "Image": linear_container,
    "ModelDataUrl": linear.model_data
}
linear_model_info = session.sagemaker_client.create_model(
                                ModelName = linear_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = linear_primary_container)

## First endpoint configuration and deployment: Two models combined

combined_endpoint_config_name = "boston-combined-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# We define an endpoint with two variants = models
# We assign a weight to each variant 
# to specify the amount of data that will go to each model.
# For instance, if we have two `ProductionVariants`
# each with `InitialVariantWeight = 1`,
# the data will be distributed in half, randomly.
combined_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = combined_endpoint_config_name,
                            ProductionVariants = [
                                { # First we include the linear model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": linear_model_name,
                                    "VariantName": "Linear-Model"
                                }, { # And next we include the xgb model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": xgb_model_name,
                                    "VariantName": "XGB-Model"
                                }])

endpoint_name = "boston-updated-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

combined_endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = combined_endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

## Use first endpoint

# We see that each sample is scored
# by a different model
# with probability 50% each
for rec in range(10):
    response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[rec])))
    print(response)
    result = response['Body'].read().decode("utf-8")
    print(result)
    print(Y_test.values[rec])

## Second endpoint configuration: One model

xgb_endpoint_config_name = "boston-update-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
xgb_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = xgb_endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": xgb_model_name,
                                "VariantName": "XGB-Model"
                            }])

## Update first endpoint with configuration of second

# IMPORTANT: For the user, the endpoint interface (URL)
# remains to be the same, but internally it's changed,
# because another set of models is started.

# Get info: first endpoint
print(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))

# We take the endpoint with two variants
# and update its configuration
# to be the one of the endpoint with the XGBoost
session.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=xgb_endpoint_config_name)

# We print the endpoint info: 
# it hasn't changed from two variants to one yet
# because internally an new endpoint is being launched.
# After launched, the data will be routed to it and the old endpoint
# with two variant will be destroyed.
print(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))

# Wait until it is updated
endpoint_dec = session.wait_for_endpoint(endpoint_name)

# Now, we have one variant!
print(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))

## Shut down endpoint

session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)

## Clean up

!rm $data_dir/*
!rmdir $data_dir

########################
### 14. Using the Endpoint from Outside: Lambda + API Gateway + Web App
########################

# The code in this section contains a very simple
# lambda function prototype, but many more things than this snippet
# need to be prepared:
# - Create IAM role for Lambda
# - Create a Lambda function
# - Create an API Gateway (REST) with POST method and Lambda proxy integration
# - HTML web app with interaction to API Gateway URL
#
# For more information, check
# Section 3.3 from DLND_Deployment.md
# https://github.com/mxagar/deep_learning_udacity/blob/main/06_Deployment/DLND_Deployment.md

# Key insights:
# - Lambda functions are functions as a service: code that is executed without caring about the server.
# - API Gateways enable REST APIs that can be accessed from the internet.
# - SageMaker endpoints cannot be accessed from outside, but they can be invoked from AWS (e.g., from API Gateway) using boto3.

# Structure of a Lambda function:
# - All imports necessary.
# - The definition of all functions and variables used.
# - `lambda_handler()`: the lambda function executed; it has an `event` object which contains the information used inside the function -- in our case, that's the plain text from the API Gateway.

############### Lambda Function ###############

# This example is the one of the sentiment analysis model
# which performs these tasks:
# - Definitions: vocabulary
# - Take text in event['body'] from API Gateway and pre-process it: review_to_words()
# - Create a Bag of Words (BOW) of the text using the vocabulary
# - Invoke SageMaker endpoint and pass serialized BoW
# - Catch result from SageMaker endpoint and pack as response for API Gateway

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
