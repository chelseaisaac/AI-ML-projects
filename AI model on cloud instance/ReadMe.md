<h1>Deploying a Simple AI Model on a Cloud GPU instance</h1>

**Objective:** 

The idea for this project is to deploy a pre-trained AI model on a GPU-enabled cloud instance (e.g., AWS EC2 with GPU) using Docker. It's a simple but functional deployment to work with GPU instances and Docker for AI workloads. 

Prerequisite:
- You wil need to request a vCPU quota increase in your AWS account in your region. Your limit usually starts at 0 and you need a least 4 to use a GPU-enabled EC2 instance.
- You will need an account for the NGC registry and your API credentials. 

<h2>1. Launch a GPU-enabled instance on a cloud platform</h2>
  
The first thing we have to do is **launch a GPU-enabled instance on a cloud platform**. I went with an AWS EC2 AMI with an GPU support, Deep Learning AMI (Ubuntu) with instance type 'g4dn.xlarge'. It's cost-effective for this simple project. (Hint: I like to use the *Pricing Calculator* to estimate the monthly charges for my services).

<h2>2. Install Docker</h2>

Once we do that, we install Docker in our instance. 
We SSH into our instance and install Docker to our instance.

Update the package list:

<code>sudo apt-get update</code>

Install Docker:

<code>sudo apt-get install docker.io -y</code>


Add the user to the docker group:

<code>sudo usermod -aG docker ${USER}</code>

My username is *ubuntu* and so for me, it would be: 

<code>sudo usermod -aG docker ubuntu</code>

*I had to reboot my instance for the changes to take effect.* 

To test the Docker installation, we simply run the following code:

<code>docker --version</code>

<h2>3. Pull a pre-trained model</h2>
Now, let's pull a pre-trained model. I went on NVIDIA NGC website to select a model. I went with Resnet50 since it's a solid choice for image classification tasks.

First log into NGC registry: 
<code>docker login nvcr.io</code>

The username is always $oauthtoken. 

Enter your API key from your NGC account.

We'll need to pull the model's Docker container from the NGC registry.

<code>docker pull nvcr.io/nvidia/pytorch:22.12-py3</code>

List the Docker images to confirm the container was pulled. 

<code>docker images</code>

<h2>4. Deploy the model</h2>
Now, let's deploy the model.
We'll run the Docker container that has GPU support.

<code>docker run --gpus all -it nvcr.io/nvidia/resnet50:latest</code>

<!-- But does it work? Let's put it to the test. 
We'll use a sample image to test the model. 

<code>python3 resnet50_inference.py --image sample.jpg</code>->





