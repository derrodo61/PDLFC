# Notes from: Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD

The following notes are taken from the free notebook

https://github.com/fastai/fastbook/tree/master

which contains the text of the book 

**"Deep Learning for Coders with fastai and PyTorc" 
by Jeremy Howard (Author) and Sylvain Gugger (Author)**

My thanks to the authors, who offer this treasure of information for free.

## Chapter 1, Intro

### Neural Networks: A Brief History

- In 1943, Warren McCulloch (neurophysiologist) and Walter Pitts (logician) developed a mathematical model of an artificial neuron in their paper "A Logical Calculus of the Ideas Immanent in Nervous Activity."
  - They asserted that neural events and their relations can be treated with propositional logic.
  - They created a simplified model of a real neuron using addition and thresholding.

- Walter Pitts was largely self-taught and received an offer to study at Cambridge University at age 12, which he declined.
  - He worked mainly while homeless, and his work with McCulloch influenced psychologist Frank Rosenblatt.

- Frank Rosenblatt further developed the artificial neuron and built the first device using these principles, the Mark I Perceptron.
  - Rosenblatt claimed this machine could perceive, recognize, and identify its surroundings without human training.

- Marvin Minsky and Seymour Papert wrote "Perceptrons," showing that a single layer of perceptrons could not learn some critical functions (e.g., XOR).
  - They also showed that multiple layers could address these limitations, but only the first insight was widely recognized, causing the academic community to abandon neural networks for two decades.

- The multi-volume "Parallel Distributed Processing (PDP)" by David Rumelhart, James McClellan, and the PDP Research Group was pivotal in neural network research.
  - The PDP approach suggested that brains' computational architecture was better suited for natural information processing tasks than traditional computer programs.
  - The PDP framework is similar to today's neural networks, requiring:
    - A set of processing units
    - A state of activation
    - An output function for each unit
    - A pattern of connectivity among units
    - A propagation rule for propagating activities through the network
    - An activation rule for combining inputs with the current state to produce an output
    - A learning rule for modifying patterns of connectivity by experience
    - An environment for the system to operate in

- In the 1980s, models with a second layer of neurons were built, avoiding problems identified by Minsky and Papert.
  - Neural networks were used for practical projects in the '80s and '90s, but theoretical misunderstandings held back the field.

- Adding more layers of neurons, as shown 30 years ago, is essential for practical good performance.
  - Recent advances in computer hardware, data availability, and algorithmic tweaks have allowed neural networks to be trained faster and more efficiently.
  - Neural networks now fulfill their potential, capable of perceiving, recognizing, and identifying surroundings without human training.

- This book will teach you how to build such systems.
  - The journey will begin with getting to know each other and understanding the foundational concepts.

### The Software: PyTorch, fastai, and Jupyter

- Fast.ai has completed numerous machine learning projects using various packages and programming languages.
- They have written courses on most major deep learning and machine learning packages.
- After PyTorch's release in 2017, fast.ai decided to adopt it for future courses, software development, and research.
- PyTorch has become the fastest-growing deep learning library and is widely used in top research papers.
- PyTorch is praised for its flexibility, expressiveness, and balance of speed and simplicity.
- PyTorch serves well as a low-level foundation library, with fastai providing higher-level functionality on top.
- The fastai library, particularly version 2, offers unique features and a layered software architecture.
- The book will progressively delve into deep learning foundations and fastai's layers.
- Emphasis is placed on learning deep learning techniques properly rather than focusing on specific software.
- The rapid evolution of deep learning libraries necessitates a focus on foundational understanding and adaptability.
- By the book's end, readers will understand the inner workings of fastai and much of PyTorch.
- Practical learning through coding and experimentation is crucial, facilitated by the Jupyter Notebook platform.
- Jupyter is the preferred tool for data science in Python due to its power, flexibility, and ease of use.

### What is Machine Learning?

- **Deep Learning Models**: Use neural networks, originating from the 1950s, recently becoming powerful due to advancements.
- **Deep Learning and Machine Learning**: Deep learning is a modern area within the broader discipline of machine learning.
- **Machine Learning vs. Regular Programming**:
  - Regular programming: Involves writing exact steps for the computer to follow.
  - Machine learning: Involves training a model to solve problems by showing it examples, allowing it to learn from data.
- **Arthur Samuel's Contribution**:
  - Introduced the concept of machine learning in 1949.
  - In his 1962 essay, he highlighted the difficulty of manually programming every step for computers.
  - Proposed showing computers examples to solve problems instead of manual programming.
  - Demonstrated effectiveness with his checkers-playing program, which learned and improved autonomously.
- **Key Concepts of Machine Learning**:
  - **Weight Assignment**: Variables that define how a model operates.
  - **Performance Testing**: An automatic means of testing the effectiveness of weight assignments.
  - **Mechanism for Improvement**: An automatic process for adjusting weights to enhance performance.
- **Model Training**:
  - Involves adjusting weights based on performance.
  - Automated mechanisms update weights to improve the model.
  - Once trained, the model uses fixed weights and operates like a regular program.
- **Learning Process**: Training involves iterative adjustments and testing until the model achieves optimal performance.
- **Machine Learning Definition**: Training of programs through experience and data rather than manual coding of each step.
### What Is a Neural Network?

- A neural network is a highly flexible machine learning model.
- It can solve a wide range of problems by adjusting its weights.
- The universal approximation theorem proves this flexibility.
- Stochastic gradient descent (SGD) is used to train neural networks by automatically updating weights.
- Neural networks and SGD mainly involve addition and multiplication operations.
- Neural networks can handle tasks like image classification:
  - Inputs: Images
  - Weights: Adjusted within the network
  - Outputs: Identified objects (e.g., "dog" or "cat")
- Model performance is measured by its accuracy in making correct predictions.
- SGD optimizes weight assignments to enhance accuracy.
- This approach aligns with Samuel's vision of machine learning models improving through experience without needing manual intervention.

### Terminology

The following is a quote:

Terminology for all the pieces we have discussed:

* The functional form of the model is called its architecture (but be careful—sometimes people use model as a synonym of architecture, so this can get confusing).
* The weights are called parameters.
* The predictions are calculated from the independent variable, which is the data not including the labels.
* The results of the model are called predictions.
* The measure of performance is called the loss.
* The loss depends not only on the predictions, but also the correct labels (also known as targets or the dependent variable); e.g., "dog" or "cat."

### Deep Learning Vocabulary (quote)

| Term             | Meaning                                                                                                 |
|------------------|---------------------------------------------------------------------------------------------------------|
| **Label**        | The data that we're trying to predict, such as "dog" or "cat"                                           |
| **Architecture** | The _template_ of the model that we're trying to fit; the actual mathematical function that we're passing the input data and parameters to |
| **Model**        | The combination of the architecture with a particular set of parameters                                 |
| **Parameters**   | The values in the model that change what task it can do, and are updated through model training          |
| **Fit**          | Update the parameters of the model such that the predictions of the model using the input data match the target labels |
| **Train**        | A synonym for _fit_                                                                                     |
| **Pretrained model** | A model that has already been trained, generally using a large dataset, and will be fine-tuned       |
| **Fine-tune**    | Update a pretrained model for a different task                                                          |
| **Epoch**        | One complete pass through the input data                                                                |
| **Loss**         | A measure of how good the model is, chosen to drive training via SGD                                     |
| **Metric**       | A measurement of how good the model is, using the validation set, chosen for human consumption           |
| **Validation set** | A set of data held out from training, used only for measuring how good the model is                   |
| **Training set** | The data used for fitting the model; does not include any data from the validation set                   |
| **Overfitting**  | Training a model in such a way that it _remembers_ specific features of the input data, rather than generalizing well to data not seen during training |
| **CNN**          | Convolutional neural network; a type of neural network that works particularly well for computer vision tasks |
