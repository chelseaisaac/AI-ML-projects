<h1>Deploying a Simple AI Model on a Cloud GPU instance</h1>

**Objective:** 

The idea for this project is to deploy a pre-trained AI model on a GPU-enabled cloud instance (e.g., AWS EC2 with GPU) using Docker. It's a simple but functional deployment to work with GPU instances and Docker for AI workloads. 

Prerequisite:
- You wil need to request a vCPU quota increase in your AWS account in your region. Your limit usually starts at 0 and you need a least 4 to use a GPU-enabled EC2 instance.
- You will need an account for the NGC registry and your API credentials. 

<h2>1. Launch a GPU-enabled instance on a cloud platform</h2>
  
The first thing we have to do is **launch a GPU-enabled EC2 instance on AWS**. I went with an AWS EC2 AMI with GPU support, Deep Learning AMI (Ubuntu) with instance type 'g4dn.xlarge'. It's cost-effective for this simple project. (Hint: I like to use the *Pricing Calculator* to estimate the monthly charges for my services). 

![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/AI%20model%20on%20cloud%20instance/Ubuntu%20AMI.png?raw=true)
![alt text](https://github.com/chelseaisaac/AI-ML-projects/blob/main/AI%20model%20on%20cloud%20instance/EC2%20instance%20type.png?raw=true)

<h2>2. Install Docker</h2>

Once we do that, we install Docker in our instance. 
We SSH into our instance and install Docker to our instance.

Update the package list:

<code>sudo apt-get update</code>

Install Docker:

<code>sudo apt-get install docker.io -y</code>

Intall NVIDIA utilities:

<code>sudo apt install nvidia-utils-470 </code>

Install NVIDIA CUDA Toolkit:

<code>sudo apt install nvidia-cuda-toolkit</code>


Add the user to the docker group:

<code>sudo usermod -aG docker ${USER}</code>


My username is *ubuntu* and so for me, it would be: <code>sudo usermod -aG docker ubuntu</code>

*I had to reboot my instance for the changes to take effect.* 

To test the Docker installation, we simply run the following code:

<code>docker --version</code>

<h2>3. Pull a pre-trained model</h2>
Now, let's pull a pre-trained model. I went on NVIDIA NGC website to select a model. I went with Resnet50 since it's a solid choice for image classification tasks.

First log into NGC registry: 
<code>docker login nvcr.io</code>

The username is always <code>$oauthtoken</code>. 

Enter your API key from your NGC account.

We'll need to pull the model's Docker container from the NGC registry.

<code>docker pull nvcr.io/nvidia/pytorch:24.08-py3</code>

*At this point, my EC2 instance ran out of EBS storage. I guess the initial amount I set up of 8GB was not enough so I increased it to 40GB.*


List the Docker images to confirm the container was pulled. 

<code>docker images</code>

At this point, if Docker is unable to find the NVIDIA GPU driver, we'll need to install it. Use <code>nvidia-smi</code> to check. 

Install the NVIDIA Container Toolkit. 
Add the NVIDIA GPG key
<code>curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg</code>

Add the repository

```
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list</code>
```

Update package list and install

<code>sudo apt-get update</code>

<code>sudo apt-get install -y nvidia-container-toolkit</code>

Configure Docker to use the NVIDIA runtime and restart it. 

<code>sudo nvidia-ctk runtime configure --runtime=docker</code>

<code>sudo systemctl restart docker</code>

Verify installation:
<code>docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi</code>

If it still doesn't work, upgrade your system:
<code>sudo apt update && sudo apt upgrade -y</code>

Install the required drivers:
<code>sudo apt install -y linux-headers-$(uname -r) build-essential</code>

Then, let's install the NVIDIA drivers used by Ubuntu:
<code>sudo apt install -y nvidia-driver-535</code>

If it freezes during the progress, restart the instance.


<h2>4. Deploy the model</h2>
Now, let's deploy the model.
We'll run the Docker container that has GPU support.

<code>docker run --gpus all -it nvcr.io/nvidia/pytorch:24.08-py3</code>

But does it work? Let's put it to the test. 
We'll use this sample image to test the model and see if it can predict the correct breed of an animal, say this really cute Samoyed dog for example.

![alt text](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg)


Create a test script
<code>cat << EOF > test_resnet50.py</code>
```
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Download and prepare a sample image
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare the image for the model
input_tensor = transform(img)
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to GPU if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Make a prediction
with torch.no_grad():
    output = model(input_batch)

# Print the top 5 predicted classes
_, indices = torch.sort(output, descending=True)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
print([(idx.item(), percentage[idx].item()) for idx in indices[0][:5]])

EOF

```

Run the test script:

<code>python test_resnet50.py</code>

So my output was this:

<code>[(258, 87.32960510253906), (259, 3.027086019515991), (270, 1.9671134948730469), (261, 1.1073544025421143), (248, 0.9204240441322327)]</code>

**Translation**: This model is highly confident, at nearly 87.33%, that the input belongs to class 258. Checking IMAGENET, class 258 is a Samoyed. 

It works!








