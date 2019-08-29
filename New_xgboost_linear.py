#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all the needed libraries
import pandas as pd
import numpy as np
from datetime import datetime
import io
import sagemaker.amazon.common as smac

import boto3
from sagemaker import get_execution_role
import sagemaker

# Importing matplot and seaborn library for visualization

import matplotlib.pyplot as plt
import seaborn as sns


# # Step 1: Loading the data from Amazon S3
# Let's get the UFO sightings data that is stored in S3 and load it into memory.

# In[4]:



role = get_execution_role()
bucket='ml-lab-amritha'
sub_folder = 'ufo-dataset'
data_key = 'ufo_fullset.csv'
data_location = 's3://{}/{}/{}'.format(bucket, sub_folder, data_key)

df = pd.read_csv(data_location, low_memory=False)
df.head()


# # Step 2: Cleaning, transforming, analyize, and preparing the dataset
# This step is so important. It's crucial that we clean and prepare our data before we do anything else.

# In[5]:



# Let's check to see if there are any missing values
missing_values = df.isnull().values.any()
if(missing_values):
    display(df[df.isnull().any(axis=1)])


# In[6]:


df['shape'].value_counts()


# In[7]:



# Replace the missing values with the most common shape
df['shape'] = df['shape'].fillna(df['shape'].value_counts().index[0])


# Let's go ahead and start preparing our dataset by transforming some of the values into the correct data types. Here is what we are going to take care of.
# 
# Convert the reportedTimestamp and eventDate to a datetime data types.
# Convert the shape and weather to a category data type.
# Map the physicalEvidence and contact from 'Y', 'N' to 0, 1.
# Convert the researchOutcome to a category data type (target attribute).

# In[9]:



df['reportedTimestamp'] = pd.to_datetime(df['reportedTimestamp'])
df['eventDate'] = pd.to_datetime(df['eventDate'])

df['shape'] = df['shape'].astype('category')
df['weather'] = df['weather'].astype('category')

df['physicalEvidence'] = df['physicalEvidence'].replace({'Y': 1, 'N': 0})
df['contact'] = df['contact'].replace({'Y': 1, 'N': 0})

df['researchOutcome'] = df['researchOutcome'].astype('category')


# In[10]:


df.dtypes


# 
# Let's visualize some of the data to see if we can find out any important information.

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("paper", font_scale=1.4)


# In[12]:


m_cts = (df['contact'].value_counts())
m_ctsx = m_cts.index
m_ctsy = m_cts.get_values()
f, ax = plt.subplots(figsize=(5,5))

sns.barplot(x=m_ctsx, y=m_ctsy)
ax.set_title('UFO Sightings and Contact')
ax.set_xlabel('Was contact made?')
ax.set_ylabel('Number of Sightings')
ax.set_xticklabels(['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()


# In[14]:


m_cts = (df['physicalEvidence'].value_counts())
m_ctsx = m_cts.index
m_ctsy = m_cts.get_values()
f, ax = plt.subplots(figsize=(5,5))

sns.barplot(x=m_ctsx, y=m_ctsy)
ax.set_title('UFO Sightings and Physical Evidence')
ax.set_xlabel('Was there physical evidence?')
ax.set_ylabel('Number of Sightings')
ax.set_xticklabels(['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()


# In[15]:


m_cts = (df['shape'].value_counts())
m_ctsx = m_cts.index
m_ctsy = m_cts.get_values()
f, ax = plt.subplots(figsize=(9,5))

sns.barplot(x=m_ctsx, y=m_ctsy)
ax.set_title('UFO Sightings by Shape')
ax.set_xlabel('UFO Shape')
ax.set_ylabel('Number of Sightings')
plt.xticks(rotation=45)
plt.show()


# In[16]:


m_cts = (df['weather'].value_counts())
m_ctsx = m_cts.index
m_ctsy = m_cts.get_values()
f, ax = plt.subplots(figsize=(5,5))

sns.barplot(x=m_ctsx, y=m_ctsy)
ax.set_title('UFO Sightings by Weather')
ax.set_xlabel('Weather')
ax.set_ylabel('Number of Sightings')
plt.xticks(rotation=45)
plt.show()


# In[17]:



m_cts = (df['researchOutcome'].value_counts())
m_ctsx = m_cts.index
m_ctsy = m_cts.get_values()
f, ax = plt.subplots(figsize=(5,5))

sns.barplot(x=m_ctsx, y=m_ctsy)
ax.set_title('UFO Sightings and Research Outcome')
ax.set_xlabel('Research Outcome')
ax.set_ylabel('Number of Sightings')
plt.xticks(rotation=45)
plt.show()


# In[18]:


ufo_yr = df['eventDate'].dt.year  # series with the year exclusively

## Set axes ##
years_data = ufo_yr.value_counts()
years_index = years_data.index  # x ticks
years_values = years_data.get_values()

## Create Bar Plot ##
plt.figure(figsize=(15,8))
plt.xticks(rotation = 60)
plt.title('UFO Sightings by Year')
plt.ylabel('Number of Sightings')
plt.xlabel('Year')

years_plot = sns.barplot(x=years_index[:60],y=years_values[:60])


# In[19]:


# To see the correlation between values
df.corr()


# Let's drop the columns that are not important.
# 
# We can drop sighting becuase it is always 'Y' or Yes.
# Let's drop the firstName and lastName becuase they are not important in determining the researchOutcome.
# Let's drop the reportedTimestamp becuase when the sighting was reporting isn't going to help us determine the legitimacy of the sighting.
# We would need to create some sort of buckets for the eventDate and eventTime, like seasons for example, but since the distribution of dates is pretty even, let's go ahead and drop them.

# In[20]:


df.drop(columns=['firstName', 'lastName', 'sighting', 'reportedTimestamp', 'eventDate', 'eventTime'], inplace=True)


# In[21]:


df.head()


# 
# Let's apply one-hot encoding
# 
# We need to one-hot both the weather attribute and the shape attribute.
# We also need to transform or map the researchOutcome (target) attribute into numeric values. This is what the alogrithm is expecting. We can do this by mapping unexplained, explained, and probable to 0, 1, 2.

# In[22]:



# Let's one-hot the weather and shape attribute
df = pd.get_dummies(df, columns=['weather', 'shape'])

# Let's replace the researchOutcome values with 0, 1, 2 for Unexplained, Explained, and Probable
df['researchOutcome'] = df['researchOutcome'].replace({'unexplained': 0, 'explained': 1, 'probable': 2})


# In[23]:


display(df.head())
display(df.shape)


# Let's randomize and split the data into training, validation, and testing.
# 
# First we need to randomize the data.
# Next Let's use 80% of the dataset for our training set.
# Then use 10% for validation during training.
# Finally we will use 10% for testing our model after it is deployed.

# In[24]:



# Let's go ahead and randomize our data.
df = df.sample(frac=1).reset_index(drop=True)

# Next, Let's split the data into a training, validation, and testing.
rand_split = np.random.rand(len(df))
train_list = rand_split < 0.8                       # 80% for training
val_list = (rand_split >= 0.8) & (rand_split < 0.9) # 10% for validation
test_list = rand_split >= 0.9                       # 10% for testing

 # This dataset will be used to train the model.
data_train = df[train_list]

# This dataset will be used to validate the model.
data_val = df[val_list]

# This dataset will be used to test the model.
data_test = df[test_list]


# 
# Next, let's go ahead and rearrange our attributes so the first attribute is our target attribute researchOutcome. This is what AWS requires and the XGBoost algorithms expects. You can read all about it here in the documentation.
# 
# After that we will go ahead and create those files on our Notebook instance (stored as CSV) and then upload them to S3.

# In[25]:


# Simply moves the researchOutcome attribute to the first position before creating CSV files
pd.concat([data_train['researchOutcome'], data_train.drop(['researchOutcome'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([data_val['researchOutcome'], data_val.drop(['researchOutcome'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)

# Next we can take the files we just stored onto our Notebook instance and upload them to S3.
boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/xgboost_train/train.csv').upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/xgboost_validation/validation.csv').upload_file('validation.csv')


# # Step 3: Creating and training our model (XGBoost)
# This is where the magic happens. We will get the ECR container hosted in ECR for the XGBoost algorithm.

# In[26]:


# This is where the magic happens. We will get the ECR container hosted in ECR for the XGBoost algorithm.

from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'xgboost')


# 
# Next, because we're training with the CSV file format, we'll create inputs that our training function can use as a pointer to the files in S3, which also specify that the content type is CSV.

# In[27]:



s3_input_train = sagemaker.s3_input(s3_data='s3://{}/algorithms_lab/xgboost_train'.format(bucket), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/algorithms_lab/xgboost_validation'.format(bucket), content_type='csv')


# 
# Next we start building out our model by using the SageMaker Python SDK and passing in everything that is required to create a XGBoost model.
# 
# First I like to always create a specific job name.
# 
# Next, we'll need to specify training parameters.
# 
# The xgboost algorithm container
# The IAM role to use
# Training instance type and count
# S3 location for output data/model artifact
# XGBoost Hyperparameters
# Finally, after everything is included and ready, then we can call the .fit() function which specifies the S3 location for training and validation data.

# In[28]:



# Create a training job name
job_name = 'ufo-xgboost-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
print('Here is the job name {}'.format(job_name))

# Here is where the model artifact will be stored
output_location = 's3://{}/algorithms_lab/xgboost_output'.format(bucket)


# In[29]:


sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.xlarge',
                                    output_path=output_location,
                                    sagemaker_session=sess)

xgb.set_hyperparameters(objective='multi:softmax',
                        num_class=3,
                        num_round=100)

data_channels = {
    'train': s3_input_train,
    'validation': s3_input_validation
}
xgb.fit(data_channels, job_name=job_name)


# In[30]:


print('Here is the location of the trained XGBoost model: {}/{}/output/model.tar.gz'.format(output_location, job_name))


# 
# After we train our model we can see the default evaluation metric in the logs. The merror is used in multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases). We want this to be minimized (so we want this to be super small).

# # Step 4: Creating and training our model (Linear Learner)
# Let's evaluate the Linear Learner algorithm as well. Let's go ahead and randomize the data again and get it ready for the Linear Leaner algorithm. We will also rearrange the columns so it is ready for the algorithm (it expects the first column to be the target attribute)

# In[31]:


np.random.seed(0)
rand_split = np.random.rand(len(df))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

 # This dataset will be used to train the model.
data_train = df[train_list]

# This dataset will be used to validate the model.
data_val = df[val_list]

# This dataset will be used to test the model.
data_test = df[test_list]

# This rearranges the columns
cols = list(data_train)
cols.insert(0, cols.pop(cols.index('researchOutcome')))
data_train = data_train[cols]

cols = list(data_val)
cols.insert(0, cols.pop(cols.index('researchOutcome')))
data_val = data_val[cols]

cols = list(data_test)
cols.insert(0, cols.pop(cols.index('researchOutcome')))
data_test = data_test[cols]

# Breaks the datasets into attribute numpy.ndarray and the same for target attribute.  
train_X = data_train.drop(columns='researchOutcome').as_matrix()
train_y = data_train['researchOutcome'].as_matrix()

val_X = data_val.drop(columns='researchOutcome').as_matrix()
val_y = data_val['researchOutcome'].as_matrix()

test_X = data_test.drop(columns='researchOutcome').as_matrix()
test_y = data_test['researchOutcome'].as_matrix()


# Next, Let's create recordIO file for the training data and upload it to S3.

# In[32]:


train_file = 'ufo_sightings_train_recordIO_protobuf.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/linearlearner_train/{}'.format(train_file)).upload_fileobj(f)
training_recordIO_protobuf_location = 's3://{}/algorithms_lab/linearlearner_train/{}'.format(bucket, train_file)
print('The Pipe mode recordIO protobuf training data: {}'.format(training_recordIO_protobuf_location))


# Let's create recordIO file for the validation data and upload it to S3

# In[33]:


validation_file = 'ufo_sightings_validatioin_recordIO_protobuf.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object('algorithms_lab/linearlearner_validation/{}'.format(validation_file)).upload_fileobj(f)
validate_recordIO_protobuf_location = 's3://{}/algorithms_lab/linearlearner_validation/{}'.format(bucket, validation_file)
print('The Pipe mode recordIO protobuf validation data: {}'.format(validate_recordIO_protobuf_location))


# 
# Alright we are good to go for the Linear Learner algorithm. Let's get everything we need from the ECR repository to call the Linear Learner algorithm.

# In[34]:



from sagemaker.amazon.amazon_estimator import get_image_uri
import sagemaker

container = get_image_uri(boto3.Session().region_name, 'linear-learner', "1")


# In[35]:


# Create a training job name
job_name = 'ufo-linear-learner-job-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
print('Here is the job name {}'.format(job_name))

# Here is where the model-artifact will be stored
output_location = 's3://{}/algorithms_lab/linearlearner_output'.format(bucket)


# Next we start building out our model by using the SageMaker Python SDK and passing in everything that is required to create a Linear Learner model.
# 
# First I like to always create a specific job name.
# 
# Next, we'll need to specify training parameters.
# 
# The linear-learner algorithm container
# The IAM role to use
# Training instance type and count
# S3 location for output data/model artifact
# The input type (Pipe)
# Linear Learner Hyperparameters
# Finally, after everything is included and ready, then we can call the .fit() function which specifies the S3 location for training and validation data.

# In[36]:


print('The feature_dim hyperparameter needs to be set to {}.'.format(data_train.shape[1] - 1))


# In[37]:


sess = sagemaker.Session()

# Setup the LinearLeaner algorithm from the ECR container
linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess,
                                       input_mode='Pipe')
# Setup the hyperparameters
linear.set_hyperparameters(feature_dim=22, # number of attributes (minus the researchOutcome attribute)
                           predictor_type='multiclass_classifier', # type of classification problem
                           num_classes=3)  # number of classes in out researchOutcome (explained, unexplained, probable)


# Launch a training job. This method calls the CreateTrainingJob API call
data_channels = {
    'train': training_recordIO_protobuf_location,
    'validation': validate_recordIO_protobuf_location
}
linear.fit(data_channels, job_name=job_name)


# In[38]:



print('Here is the location of the trained Linear Learner model: {}/{}/output/model.tar.gz'.format(output_location, job_name))


# In[ ]:




