<h1>Deploying a Scalable Deep Learning Model on AWS Cloud using NVIDIA AI Enterprise</h1>

🛠️ **Tech & Skills:** 
- **Cloud Platforms:** AWS
- **NVIDIA Tools:** NVIDIA AI Enterprise, Triton Inference Server, GPU Optimization
- **DevOps:** Docker, Kubernetes
- **Programming:** Python, REST APIs
- **MLOps:** Model deployment, scaling, monitoring

**Objective:** 

The idea behind this project is to deploy a deep learning model on AWS using cloud-native technologies and NVIDIA AI Enterprise tools.


**Prerequisite:**
- You wil need to request a vCPU quota increase in your AWS account in your region. Your limit usually starts at 0 and you need a least 4 to use a GPU-enabled EC2 instance.

# 1. AWS Cloud Setup

The first thing we have to do is **launch a GPU-enabled EC2 instance on AWS**. I went with an AWS EC2 AMI with GPU support, Deep Learning AMI (Ubuntu) with instance type 'g4dn.xlarge'. Make sure to set the storage size to at least 50 GB. I went with 150 GB. 

Once our instance is created, let's connect to it via SSH. We'll need to verify the CUDA and NVIDIA GPU drivers are installed. 

Update the package list:

<code>sudo apt-get update</code>

Verify the latest NVIDIA GPU driver is installed:

<code>nvidia-smi</code>

Verify CUDA Toolkit is installed correctly: 

<code>nvcc --version</code>

Our Deep Learning AMI includes CUDA so, no surprise, it's installed.

Verify cuDNN is installed. *cuDNN is a nice GPU-accelerated library of primitives for deep neural networks (neurons, layers, weights, loss function, optiimzation algorithm, etc).* 

<code>nvcc --version</code>

If it's not, you can just run this command: 

<code>sudo apt-get install -y libcudnn8 libcudnn8-dev</code>

*At this point, I realized I wanted to be able to access Jupyter notebooks so I updated my security group rules to allow ports 8888,8786,8787 to be able to open it from a web browser. We'll also need port 8000 for our Triton server.* 

![alt text]()

No need to restart the EC2 instance as the changes to security group rules go into effect immediately.

We'll also install AWS EFS or FSx to provide persistent storage for our Kubernetes cluster. 

Create an EKS cluster. 

Ensure GPU support is enabled by installing NVIDIA device plugin for Kubernetes:

<code>kubectl apply -f https://github.com/NVIDIA/k8s-device-plugin/blob/master/nvidia-device-plugin.yml</code>

Configure node groups with GPU instances to serve as the worker nodes in your cluster.


# Let's install our NVIDIA AI Enterprise Tools

Install **NVIDIA Triton Inference Server**. *We'll use it to optimize and serve our machine learning model.*

<code>docker pull nvcr.io/nvidia/tritonserver:22.12-py3</code>

Let's deploy a simple model on Triton Inference Server and run a test inference to validate the setup:

<code>docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:22.12-py3</code>

My container does not have a <code>/models</code> directory so I tried starting the server with a flag pointing to an empty directory. 

<code>docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/tmp</code>

Success! 

Install **RAPIDS for Data Science Workloads**. *It'll give us GPU-accelerated data science libraries.*

<code>docker pull rapidsai/notebooks:24.10a-cuda12.5-py3.11</code>

```
docker run --gpus all --rm -it \
    --shm-size=1g --ulimit memlock=-1 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/notebooks:24.10a-cuda12.5-py3.11
```

Lets install Miniconda. It comes prepackaged with python and other useful tools. 

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Verify installation: 

<code>conda --version</code>

Let's create a new environment for Jupyter notebooks:

<code>conda create -n jupyter_env python=3.8</code>

Activate the environment:

<code>conda activate jupyter_env</code>

Install Jupyter:

<code>conda install jupyter</code>

Launch Jupyter

<code>jupyter notebook --ip 0.0.0.0 --no-browser --port 8888</code>

Navigate to <code>http://(EC2 Public IP):8888/?token=<some_long_string_of_characters></code> in your web browser to access Jupyter. 

![alt text]()

*You'll have to look in the terminal to find the token to access Jupyter.*

<!-- You can use <code>conda deactivate</code> to exit the environment. -->

Install **TensorRT** for inference acceleration. We'll use it to take our deep learning model and make predictions quickly and efficiently. <!-- *TensorRT is an ecosystem of APIs for high-performance deep learning inference.* -->

<code>sudo apt install tensorrt</code>

Verify it was succesfully installed here: 

<code>dpkg -l | grep tensorrt</code>


# 2. Choose a pre-trained model

Let's choose a pre-trained model from TensorFlow Hub. We'll go with BERT since it's good for natural language processing. We'll use it to do some sentiment analysis. (Ex. *Was this review positive or negative?* Things like that.)


We'll install PyTorch and Transformers. 

<code>pip install torch transformers datasets</code>

Update the Accelerate library:

<code>pip install accelerate -U</code>

We'll use the pre-trained BERT model from Hugging Face's Transformers Library and run some python code:

```
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained BERT model and tokenizer from Hugging Face's Transformers library
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Import IMDb movie review datasets through Hugging Face's datasets library
dataset = load_dataset('imdb')

# Tokenize the training data using BERT's tokenizer (converts text into a format that BERT expects)
train_dataset = dataset['train'].map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

# Convert the tokenized datasets into PyTorch tensors (the fundamental data structure in PyTorch)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments. We'll train the model for 3 epochs
training_args = TrainingArguments(
    output_dir='./results',         # where model checkpoints and outputs will be saved
    num_train_epochs=3,
    per_device_train_batch_size=16, # each GPU will process 16 examples at a time
    save_steps=10_000,              # model saved every 10,000 training steps
    save_total_limit=2,             # only last 2 models are kept to save space
    logging_dir='./logs',           # training logs
    logging_steps=500,              # log every 500 steps
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,    # the best model wins
    metric_for_best_model="loss",   # use loss to determine best model
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset['test'].map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)
)

# Fine-tune or train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
```

*It took about 2 hours 14 mins to train my model.*

Let's upload our model and tokenizer to S3 so I can have my tuned model on hand and not have to do this again.

I created a folder in my bucket called "fine_tuned_bert".

In order to upload the directory to S3, we'll need to add an IAM role to our EC2 instance that will allow EC2 to accesss S3.

![alt text]()

Upload the directory that contains our model & tokenizer to our S3 bucket:

<code>aws s3 cp ./fine_tuned_bert s3://your-bucket-name/fine_tuned_bert/ --recursive</code>

![alt text]()

Now, let's optimize our model for GPU execution using TensorRT. 

First, let's convert our Hugging Face model to ONNX, a format TensorRT can use:

Let's first install the libraries:

<code>pip install torch onnx transformers</code>

Then we can use this Python script to convert to ONXX

```
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your model and tokenizer
model_path = './fine_tuned_bert'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Create dummy input
dummy_input = tokenizer("This is a sample text", return_tensors="pt")

# Export the model
torch.onnx.export(model,                     # model being run
                  (dummy_input.input_ids, dummy_input.attention_mask),  # model input (or a tuple for multiple inputs)
                  "bert_model.onnx",         # where to save the model
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input_ids', 'attention_mask'],   # the model's input names
                  output_names=['output'],   # the model's output names
                  dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence'},    # variable length axes
                                'attention_mask' : {0 : 'batch_size', 1: 'sequence'},
                                'output' : {0 : 'batch_size', 1: 'sequence'}})

print("Model exported to ONNX format")
```

# 3. Containerization

# 4. Deploy containerized model to a Kubernetes cluster on AWS

# 5. Inference service


