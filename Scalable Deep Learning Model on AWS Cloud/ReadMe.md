<h1>Deploying a Scalable Deep Learning Model on AWS Cloud using NVIDIA AI Enterprise</h1>

🛠️ **Tech & Skills:** 
- **Cloud Platforms:** AWS
- **NVIDIA Tools:** NVIDIA AI Enterprise, Triton Inference Server, TensorRT
- **DevOps:** Docker, Kubernetes
- **Programming:** Python, REST APIs
- **MLOps:** Model deployment, scaling, monitoring

**Objective:** 

The idea behind this project is to deploy a deep learning model on AWS using cloud-native technologies and NVIDIA AI Enterprise tools.

![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/Tuning%20BERT%20model.png?raw=true)

![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/Using%20the%20tuned%20BERT%20model.png?raw=true)

**Prerequisite:**
- You wil need to request a vCPU quota increase in your AWS account in your region. Your limit usually starts at 0 and you need a least 4 to use a GPU-enabled EC2 instance. I requested 24 vPCUs.

# 1. AWS Cloud Setup

The first thing we have to do is **launch a GPU-enabled EC2 instance on AWS** for our build environment. I went with an AWS EC2 AMI with GPU support, Deep Learning AMI (Ubuntu) with instance type 'g4dn.xlarge'. Make sure to set the storage size to at least 50 GB. I went with 150 GB. 

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


# Let's install our NVIDIA AI Enterprise Tools

Install **NVIDIA Triton Inference Server**. *We'll use it to optimize and serve our machine learning model.*

<code>docker pull nvcr.io/nvidia/tritonserver:23.10-py3</code>

Let's deploy a simple model on Triton Inference Server and run a test inference to validate the setup:

<code>docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:23.10-py3</code>

My container does not have a <code>/models</code> directory so I tried starting the server with a flag pointing to an empty directory. 

<code>docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/tmp</code>

Success! 

Install **TensorRT** for inference acceleration. We'll use it to take our deep learning model and make predictions quickly and efficiently. <!-- *TensorRT is an ecosystem of APIs for high-performance deep learning inference.* -->

<code>sudo apt install tensorrt</code>

Verify it was succesfully installed here: 

<code>dpkg -l | grep tensorrt</code>

Install TensorRT Python wheel 

<code>python3 -m pip install --upgrade tensorrt</code>

Verify that it was installed. 

```
python3
>>> import tensorrt
>>> print(tensorrt.__version__)
>>> assert tensorrt.Builder(tensorrt.Logger())
```

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

![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/1EC2%20IAM%20role%20policy.png?raw=true)

Upload the directory that contains our model & tokenizer to our S3 bucket:

<code>aws s3 cp ./fine_tuned_bert s3://your-bucket-name/fine_tuned_bert/ --recursive</code>

<!---![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/2Fine%20tuned%20model%20in%20S3%20bucket.png?raw=true)-->

Now, let's optimize our model for GPU execution using TensorRT. 

First, let's convert our Hugging Face model to ONNX, a format TensorRT can use:

Let's first install the libraries:

<code>pip install torch onnx transformers</code>

Then we can use this Python script to convert to ONXX. *We do this to to leverage acceleration tools like TensorRT.*

```
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = './fine_tuned_bert'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

# Create dummy input with dynamic sequence length
dummy_input = tokenizer("This is a sample text", return_tensors="pt")

# Export the model
torch.onnx.export(model,
                  (dummy_input.input_ids, dummy_input.attention_mask),
                  "bert_model.onnx",
                  export_params=True,
                  opset_version=14,
                  do_constant_folding=True,
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                                'output': {0: 'batch_size'}})

print("Model exported to ONNX format")
```

Now, let's optimize our model for GPU execution with TensorRT.

*A TensorRT engine is a highly optimized, hardware-specific representation of a deep learning model that processed by NVIDIA TensorRT. It converts models from common deep learning frameworks (e.g., PyTorch, TensorFlow, ONNX) into an optimized format that can run efficiently on NVIDIA GPUs.*

This Python script creates a TensorRT engine file named "bert_model.plan". You can also build your TensorRT engine engine in a Triton container to better control and manage the TensorRT versions in your build environment and container environmennt (provided under the script).

```
import tensorrt as trt
import os

def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
        # Configure the builder
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 mode if your GPU supports it
        
        # Load the ONNX file
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(explicit_batch) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            # Build the engine
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            attention_mask_name = network.get_input(1).name
            profile.set_shape(input_name, (1, 6), (8, 256), (32, 512))
            profile.set_shape(attention_mask_name, (1, 6), (8, 256), (32, 512))
            config.add_optimization_profile(profile)
            
            engine = builder.build_serialized_network(network, config)
            
            # Serialize the engine
            with open("bert_model.plan", "wb") as f:
                f.write(engine)
            
            return engine

# Optimize the model
onnx_file_path = 'bert_model.onnx'
trt_engine = build_engine(onnx_file_path)

if trt_engine:
    print("TensorRT engine built successfully")
else:
    print("Failed to build TensorRT engine")
```



*As I said before, you should build the TensorRT engine in a Triton container. Here's a Dockerfile to create the Triton container.*

```
# Use Triton Inference Server with TensorRT 10.4.0.26
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install additional dependencies such as pycuda and any others you need
RUN pip3 install --no-cache-dir pycuda onnx torch transformers

# Set up a directory for the model and copy your ONNX model into the container
WORKDIR /workspace
COPY ./your_model.onnx /workspace/your_model.onnx

# Expose Triton's ports for inference
EXPOSE 8000 8001 8002

# Default command is to open a bash shell (or you could directly launch Triton)
CMD ["/bin/bash"]

```

Build the Triton container:

<code>docker build -t my_triton_with_tensorrt_10_4 .</code>


Run the container interactively with GPU access:

<code>docker run --gpus all -it --rm my_triton_with_tensorrt_10_4 bash</code>


Inside container, convert ONXX model to a TensorRT engine:

<code>trtexec --onnx=bert_model.onnx --saveEngine=model.plan</code>


Verify the engine: 

<code>trtexec --loadEngine=model.plan</code>

*If you're not able to run the trtexec command, find where it is using* <code>sudo find / -name trtexec</code>
*Then add the directory containing it to your path: *(You can replace the path to wherever your trtexec is located).**
```
echo 'export PATH=$PATH:/usr/src/tensorrt/bin' >> ~/.bashrc
source ~/.bashrc
```

Organize the model repository in the container:
```
mkdir -p /workspace/triton-models/bert_model/1
mv /workspace/model.plan /workspace/triton-models/bert_model/1/
```

Create a <code>config.pbtxt</code> file:

```
name: "bert_model"
backend: "tensorrt"
max_batch_size: 1  # Model expects batch size of 1
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 1 ]  # Fixed sequence length of 1 (as per the model's expectation)
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 1 ]  # Fixed sequence length of 1
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 2 ]  # Adjust based on your model's output shape
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

Send your engine and <code>config.pbtxt</code> file from your Docker container to your local EC2 instance. For example, to send the engine, issue this command (*Use <code>docker ps</code> to find container id.*):

<code>docker cp <container_id>:/workspace/model.plan ./model.plan</code>

We'll save our TensorRT engine and <code>config.pbtxt</code> to S3:

<code>aws s3 cp bert_model.plan s3://your-bucket-name/triton-models/bert_model/1/model.plan</code>

<code>aws s3 cp config.pbtxt s3://your-bucket-name/triton-models/bert_model/config.pbtxt</code>

![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/3Triton%20model%20in%20EC2.png?raw=true)

In order for us to use this model with Triton, we'll create repository for Triton with this structure:

```
triton-models/
└── bert_model/
    ├── 1/
    │   └── model.plan
    └── config.pbtxt
```

To test the rebuilt engine with Triton, we'll deploy it using the Triton Inference Server:

```
docker run --rm --gpus all \
    -v /path/to/your/model/repository:/models \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models

```


# 3. Containerization
Now it's time to containerize our model. To do this, we'll create a Docker image with the optimized model, Triton Inference Server, and dependencies. *This Dockerfile assumes that you have your models saved in your local directory in a folder called "triton-models".*

```
# Use the latest official Triton Inference Server container which includes TensorRT
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install additional Python packages
RUN pip3 install --no-cache-dir \
    transformers \
    torch \
    onnx \
    pycuda

# Set up the model repository
WORKDIR /models
COPY ./triton-models /models

# Expose Triton's ports
EXPOSE 8000 8001 8002

# Start Triton Inference Server
CMD ["tritonserver", "--model-repository=/models"]

# Health check to ensure Triton is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/v2/health/ready || exit 1
```

Let's build the Docker image:

<code>docker build -t bert-triton-server:v1 .</code>

Now let's push container image to the Amazon ECR:

```
# Authenticate
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com

# Create repository (if needed)
aws ecr create-repository --repository-name bert-triton-server --region your-region

# Tag image
docker tag bert-triton-server:v1 <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/bert-triton-server:latest

# Push image
docker push <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/bert-triton-server:latest
```

# 4. Deploy containerized model to a Kubernetes cluster on AWS

Let's create a Kubernetes cluster in AWS using EKS:
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/4EKS%20cluster%201.png?raw=true)
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/5EKS%20cluster%202.png?raw=true)
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/6EKS%20cluster%205.png?raw=true)

I attached a role with the following policy:
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/7EKS%20cluster%20role.png?raw=true)

Now, let's configure node groups with GPU instances to serve as the worker nodes in our cluster. 
<!---![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/8EKS%20cluster%20nodes.png?raw=true) -->
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/8bEKS%20cluster%20nodes%202a.png?raw=true)
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/9EKS%20cluster%20nodes%20-3a.png?raw=true)

We'll need to attach an IAM role to our node group with necessary permissions to be able to pull docker images from ECR, access S3, CloudWatch and EFS. 
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/Scalable%20Deep%20Learning%20Model%20on%20AWS%20Cloud/10EKS%20node%20group%20role.png?raw=true)

Here's a Kubernetes deployment YAML file that specifies the Docker image from ECR and requests GPU resources:
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: triton-server
  template:
    metadata:
      labels:
        app: triton-server
    spec:
      containers:
      - name: triton-server
        image: <your-aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/your-ecr-repo/bert-triton-server:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        - containerPort: 8002
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: triton-service
spec:
  type: LoadBalancer
  selector:
    app: triton-server
  ports:
    - name: http
      protocol: TCP
      port: 8000
      targetPort: 8000
    - name: grpc
      protocol: TCP
      port: 8001
      targetPort: 8001
    - name: metrics
      protocol: TCP
      port: 8002
      targetPort: 8002
```

Now, we'll log into AWS Cloudshell to issue the following commands. 

Since we created our cluster using EKS, we need to update our kubeconfig file:

<code>aws eks --region your-region update-kubeconfig --name your-cluster-name</code>

Let's also ensure GPU support is enabled by installing the NVIDIA device plugin for Kubernetes:

<code>kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml</code>

Verify you can connect to the cluster:

<code>kubectl get nodes</code>

Since we're manually pulling images from ECR, let's authenticate our Docker client with ECR:

<code>aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<your-region>.amazonaws.com</code>

Add the deployment to the EKS cluster:

<code>kubectl apply -f triton-deployment.yaml</code>

Look for existing pods:
<code>kubectl get pods</code>

View what containers are inside the pods:
<code>kubectl describe pods</code>

*My pods had a hard time getting created. I initially got a FailedScheduling error because the nodes did not match pod's node affinity/selector. So I added a label to my GPU nodes:*
<code>kubectl label nodes add-node-name-here accelerator=nvidia-gpu</code>

<!---
Let's set up Horizontal Pod Autosaler which enables autoscaling to manage the number of pods based on GPU usage: 

<code>kubectl autoscale deployment triton-deployment --cpu-percent=70 --min=1 --max=3</code>

Then verify that the HPA was created:

<code>kubectl get hpa</code> 
-->

# 5. Inference service

To use our model once deployed, let's interact with it using the Triton Inference Server's HTTP/gRPC API or client libraries. 

First, let's get the external address to interact with our Triton server:

<code>kubectl get services</code>


Let's install Triton with HTTP support:

<code>pip install --user tritonclient[http]==2.33.0</code>

Here's a Python example using the Triton client library:


```
import tritonclient.http as httpclient
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

try:
    # Initialize the Triton HTTP client (remove the 'http://' prefix)
    client = httpclient.InferenceServerClient(url="your-triton-service-url:8000")  # No 'http://'
    
    # Check if the server is ready
    if not client.is_server_ready():
        print("Server is not ready")
        exit(1)

    # Prepare your input data (shape: [1, 1], dtype: int64)
    input_ids = np.array([[1]], dtype=np.int64)  # Example input_ids
    attention_mask = np.array([[1]], dtype=np.int64)  # Example attention_mask

    # Print the input data
    print(f"Input IDs: {input_ids}")
    print(f"Attention Mask: {attention_mask}")

    # Create the input tensors
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),  # input_ids tensor
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")  # attention_mask tensor
    ]
    
    # Set the data for the input tensors
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    # Specify the output (adjust based on your model's output tensor name)
    outputs = [httpclient.InferRequestedOutput("output")]

    # Send the inference request
    results = client.infer("bert_model", inputs, outputs=outputs)

    # Get the output as a numpy array
    output = results.as_numpy("output")

    # Apply softmax to get probabilities
    probabilities = softmax(output[0])

    # Map to sentiment labels
    sentiments = ['Negative', 'Positive']
    predicted_sentiment = sentiments[np.argmax(probabilities)]
    confidence = np.max(probabilities)

    print(f"Raw output: {output}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted sentiment: {predicted_sentiment}")
    print(f"Confidence: {confidence:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")
```

Here's the output we get:

Input IDs: [[1]]

Attention Mask: [[1]]

Raw output: [[-2.5691175  1.7277087]]

Probabilities: [0.0134289  0.98657113]

Predicted sentiment: Positive

Confidence: 0.9866

This means that the model is very confident (98.66% sure) that the input text was positive!
