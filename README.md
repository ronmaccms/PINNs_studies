<a id="readme-top"></a>
<h2>About The Folder</h2>

<h3>Learning and Development with NVIDIA Modulus Sym</h3>
<p>As part of the development process for this project, I am actively learning how to leverage NVIDIA’s <code>modulus.sym</code> for building PINNs. The training sessions required for developing accurate models often take up to 12 hours, and they are conducted on my GPU using a WSL. This setup allows me to fully utilize the computational power of my NVIDIA GPU while working in a Linux environment.</p>

<p>The 2D simulation in Section 3: Deep Neural Operator (DeepONet) demonstrates the power of integrating key equations such as the Navier-Stokes and Advection-Diffusion equations to model fluid dynamics. By utilizing <code>modulus.sym</code>, I have been able to develop neural networks that predict fluid velocity, pressure, and diffusivity, simulating wind flow in urban environments.</p>

<p>PINNs represent an innovative approach in computational modeling by embedding partial differential equations (PDEs) directly into the neural network’s training process. This ensures that the model adheres to established physical laws, such as the Navier-Stokes equations, leading to highly accurate simulations of complex systems like urban wind flow.</p>

<div align="center">
    <img src="https://github.com/ronmaccms/macadThesis24/blob/main/web-app/src/assets/doc/data/images/navier-stokes.png" alt="Navier-Stokes Equations" width="500">
</div>

<h4>Boundary and Initial Conditions</h4>
<p>The PINNs are configured with boundary conditions derived from real-time data. These conditions control the entry, exit, and flow of wind through the simulated urban landscape.</p>

<p>Integrating data from NOAA, OpenWeatherMap, and OpenStreetMap into the PINN model allows for the dynamic adaptation of simulations to real-time environmental conditions. This ensures that the model remains responsive and accurate, providing valuable insights for urban planners.</p>

<div align="center">
    <img src="https://github.com/ronmaccms/macadThesis24/blob/main/web-app/src/assets/doc/data/images/2d-heat-sink-with-fins.png" alt="Wind Simulation" width="600">
</div>

<h3>Understanding Modulus Sym Nodes</h3>

<p>In Modulus Sym, <strong>Nodes</strong> play a critical role in defining the components that will be executed during the forward pass of the training process. Here's a breakdown of how Nodes function and what they can represent:</p>

<h4>Key Aspects of Modulus Sym Nodes:</h4>

<ul>
    <li><strong>Wrappers for <code>torch.nn.Module</code>:</strong> A Node can wrap around PyTorch's <code>torch.nn.Module</code>, meaning that neural network architectures (like fully connected networks, CNNs, etc.) can be encapsulated as Nodes. This allows for greater flexibility and easy integration of deep learning models into the Modulus Sym framework.</li>
    <li><strong>Input and Output Variables:</strong> Nodes specify their required input and output variables. This information is essential because it allows Modulus Sym to understand how different components are connected. For instance, the output of one Node can be fed as the input to another Node in a computational graph.</li>
    <li><strong>Automatic Derivative Computation:</strong> Nodes in Modulus Sym automatically compute derivatives needed for physics-based simulations, such as the gradient of the output with respect to input variables. For example, if your model needs the gradient of velocity for Navier-Stokes, Modulus Sym ensures these derivatives are calculated.</li>
</ul>

<h4>Types of Nodes:</h4>

<ul>
    <li><strong>Pre-built PyTorch Networks:</strong> Use existing architectures that are part of Modulus Sym.</li>
    <li><strong>Custom PyTorch Networks:</strong> You can define and use your own custom neural networks by creating a <code>torch.nn.Module</code> and wrapping it in a Node.</li>
    <li><strong>Feature Transformations:</strong> These could be any form of preprocessing or transformations applied to your features (inputs).</li>
    <li><strong>Equations or Constraints:</strong> Nodes can also represent mathematical equations (like PDEs) that are part of the training process. This is useful in physics-informed neural networks (PINNs), where the loss function incorporates physics equations.</li>
</ul>

<h4>Example of Node Usage:</h4>
<p>In a physics-informed simulation, let's say you're using the Navier-Stokes equation to model fluid dynamics. You might define nodes for:</p>

<ul>
    <li>A neural network that approximates the velocity and pressure of the fluid.</li>
    <li>Nodes for calculating derivatives like gradients or divergence of the velocity field.</li>
    <li>Nodes for applying boundary conditions or initial conditions.</li>
</ul>

<h4>Benefits:</h4>

<ul>
    <li><strong>Graph-based Execution:</strong> The input/output relationships between nodes allow Modulus Sym to build a computational graph, streamlining the training process and allowing for more efficient derivative computations.</li>
    <li><strong>Flexibility:</strong> Since nodes can represent anything from equations to neural networks, you can easily integrate various components in your model without manual configuration of gradients or dependencies.</li>
</ul>

<h4>Automatic Derivative Calculation:</h4>
<p>One of the significant advantages of using Nodes is that Modulus Sym can automatically compute derivatives required for physics-based simulations. For example, in a fluid dynamics simulation using the Navier-Stokes equation, Modulus Sym will automatically calculate derivatives of the velocity field (e.g., gradients) as needed to ensure that the physics laws are satisfied.</p>

<p>This automatic calculation is critical because it saves you from manually coding the complex math required for differential equations, especially in the case of higher-order derivatives.</p>

---

<h3>Solver in Modulus Sym</h3>
<p>The <strong>Solver</strong> is the core engine of Modulus Sym, responsible for managing the optimization loop and ensuring that the model is trained efficiently. The Solver takes in the <strong>Domain</strong>, which contains all the <strong>Constraints</strong>, <strong>Inferencers</strong>, <strong>Validators</strong>, and <strong>Monitors</strong>, and handles the training process by executing them in the correct order.</p>

<h4>Key Aspects of the Solver:</h4>

<ul>
    <li><strong>Optimization Loop:</strong> The Solver computes the global loss based on all the Constraints in the Domain and optimizes the neural network by updating the trainable models (Nodes) accordingly.</li>
    <li><strong>Training Management:</strong> The Solver manages the interaction between the Constraints, Validators, Inferencers, and Monitors. It ensures that each component works together seamlessly to achieve the training objectives.</li>
</ul>

<h4>Example Code Snippet for Solver:</h4>

```python
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain

# Create a domain with predefined constraints and nodes
domain = Domain()

# Define the solver
solver = Solver(domain)

# Start the optimization loop
solver.train()
```
<p>In the example above, the Solver initializes the optimization process by taking the defined Domain, which contains all the Constraints (like Navier-Stokes equation, boundary conditions), and running the optimization loop until the global loss is minimized.</p>

<h3>Hydra in Modulus Sym</h3> <p><strong>Hydra</strong> is a configuration management package integrated into Modulus Sym that allows users to easily set and modify <strong>hyperparameters</strong> (parameters such as learning rate, batch size, and model architecture) using YAML configuration files. Hydra simplifies the process of managing and tuning these parameters across different runs.</p> <h4>Key Features of Hydra:</h4> <ul> <li><strong>Centralized Configuration:</strong> All hyperparameters are managed through YAML files, making it easy to tweak settings without modifying the code.</li> <li><strong>Parameter Flexibility:</strong> Hydra allows you to create different configurations for different models or experiments and switch between them easily.</li> <li><strong>Reproducibility:</strong> By saving configurations in YAML files, Hydra ensures that your experiments are reproducible, as the configuration can be shared or re-used without confusion.</li> </ul> <h4>Example Hydra Configuration File (config.yaml):</h4>

```
model:
  layers: 4
  units_per_layer: 64
  activation: relu

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 1000
```

<p>In this configuration file, we define a model with 4 layers, 64 units per layer, and a ReLU activation function. The training process uses a learning rate of 0.001 and a batch size of 32 for 1000 epochs. Hydra allows you to easily change these settings without needing to touch the core Python code.</p> <h4>Using Hydra with Modulus Sym:</h4>

```
import hydra
from omegaconf import DictConfig
from modulus.sym.solver import Solver

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig):
    solver = Solver(cfg)  # Load parameters from the Hydra configuration
    solver.train()

if __name__ == "__main__":
    main()
```
<p>In the Python script, Hydra is used to load the configuration file. The Solver is initialized with the parameters from the YAML configuration, making it easy to manage hyperparameters for various experiments.</p>

<h3>Git and WSL Cheat Sheet</h3> 
<h4>Essential Git Commands</h4> 
<p>Below are some essential Git commands for managing your repository:</p> 
<ul> 
    <li><strong>Checking for the 10 largest files in your commit:</strong> <pre><code>du -ah . | sort -rh | head -n 10</code></pre></li> <li><strong>Removing a file from your Git commit:</strong> <pre><code>git rm --cached "File Path"</code></pre></li> 
    <li><strong>Adding a file to .gitignore:</strong> <pre><code>echo "filename" >> .gitignore</code></pre></li> <li><strong>Committing and pushing changes:</strong> <pre><code>git add .gitignore git commit -m "Add large CUDA file to .gitignore" git push origin main</code></pre></li> <li><strong>Creating and managing branches:</strong> 
    <ul> 
        <li>To create a new branch: <pre><code>git checkout -b new-branch-name</code></pre></li> <li>To switch to an existing branch: <pre><code>git checkout branch-name</code></pre></li> 
        <li>To delete a branch: <pre><code>git branch -d branch-name</code></pre></li> 
        <li>To push a new branch to the remote repository: <pre><code>git push origin new-branch-name</code></pre></li> 
    </ul> 
    </li> 
</ul> 
<h4>Checking NVIDIA GPU Status</h4> 
<p>To check if your system has an NVIDIA GPU and ensure that it’s correctly configured for CUDA:</p> 
<ul> 
    <li>Run the following command in your terminal: <pre><code>nvidia-smi</code></pre></li> 
    <li>If you see your GPU details listed, your system is set up correctly.</li> 
</ul> 
<h4>Installing CUDA on Ubuntu 20.04</h4> 
<p>If you need to install CUDA, follow these steps:</p> 
<ul> 
    <li><strong>Install CUDA Toolkit:</strong> <pre><code>sudo apt-get install -y cuda</code></pre></li> 
    <li><strong>Verify CUDA installation:</strong> <pre><code>nvcc --version</code></pre></li> 
</ul> <h4>Setting Up WSL</h4> <p>To set up WSL with Ubuntu 20.04 on your system:</p> 
<ol> 
    <li><strong>Enable WSL:</strong> Open PowerShell as an Administrator and run: <pre><code>wsl --install</code></pre></li> 
    <li><strong>Install Ubuntu 20.04:</strong> <pre><code>wsl --install -d Ubuntu-20.04</code></pre></li> 
    <li><strong>Set Up WSL:</strong> Open Ubuntu from your Start menu and follow the on-screen instructions to complete the setup.</li> 
</ol> 
<h3>Running Your First Simulation</h3> 
<p>After setting up your environment, you can run your first PINN simulation by navigating to the project directory and executing the following command:</p> 
<pre><code>python3 your_simulation_script.py</code></pre> 
<p>This command will start the simulation, utilizing your NVIDIA GPU to accelerate the computations. Depending on the complexity of the model, training can take several hours.</p> 
<h3>Contributor</h3> 
<p>Andres</p> 
<p align="right">(<a href="#readme-top">back to top</a>)</p>



