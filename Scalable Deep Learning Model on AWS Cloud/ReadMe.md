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

Verify installation 

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

You can use <code>conda deactivate</code> to exit the environment. 

Install **TensorRT** for inference acceleration:

<code>sudo apt install tensorrt</code>

We verify it was succesfully installed here: 

<code>dpkg -l | grep tensorrt</code>


# 2. Choose a pre-trained model

Let's choose a pre-trained model from TensorFlow Hub. We'll go with BERT since it's good for natural language processing and use it to do some sentiment analysis. (Ex. *Was this review positive or negative?* Things like that.)


We'll install PyTorch and Transformers. 

<code>pip install torch transformers datasets</code>

We'll use the pre-trained BERT model from Hugging Face's Transformers Library and run some python code:

```
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels for sentiment
```

Now, we'll import the IMDb movie reviews datasets available through Hugging Face's datasets library:

```
from datasets import load_dataset

dataset = load_dataset("imdb")
train_dataset = dataset['train']
test_dataset = dataset['test']
```

Let's tokenize the test data using BERT's tokenizer. *It will convert the text into a format BERT expects.*

```
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
```

Now let's convert the tokenized datasets into PyTorch dataloaders to train the model. 

```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)
```

**Now, it's time to fine-tune our model.**

We start with setting up the training loop by using the AdamW optimizer, a learning rate schedule to adjust the learning rate during the training process:

```
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
```

Let's fine-tune the BERT model on the training dataset for 3 epochs:

<!--  -->

```
import torch
from torch import nn
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # 3 epochs
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
```

Let's evaluate the model on our test data to check its performance:

```
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)

        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

Let's save the fine-tuned model locally. 

```
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
```


Upload the model to our S3 bucket:

<code>aws s3 cp ./fine_tuned_bert s3://your-bucket-name/fine_tuned_bert/ --recursive</code>


# 3. Containerization

# 4. Deploy containerized model to a Kubernetes cluster on AWS

# 5. Inference service


