# Rerun (Machine Learning)
To read this repository, it is necessary to understand how the world works, what intelligence is, what it is, how data is distributed throughout the world, and how each piece of information carries weight and more. If you don't know this, you can read my repository explaining it perfectly (https://github.com/tevoshw/machine-learning)

> This repository will be more professional in focus, without analogies at all times and visual examples, I suppose you master most of mathematical concepts, and machine learning.

# 1. Deep Learning (Deep Learning)

**Deep Learning** is a subarea of Machine Learning that is based on a specific framework: **Deep Artificial Neural Networks**. While traditional ML often hits a learning "ceiling," Deep Learning has a unique characteristic: **the more data and computing power you provide, the better it gets.**

He is the technology that has allowed AI to move out of the tables and into the world of the senses (sight, hearing and language).

---

## 1.1 The Structure: Why "Deep" (Deep)?

The term "Deep" refers to the number of **Hidden Layers (Hidden Layers)** between the data input and the result output. 

* **Classic ML:** Usually works with "shallow layers" (one or two layers) or direct rules.
* **Deep Learning:** Uses tens or even thousands of layers. Each layer is responsible for learning a different level of abstraction.

> **Example Vision:** At the first layer, the network learns to identify edges. In the second, geometric shapes. In the tenth, parts of the face. At the hundredth, she understands the difference between a human face and that of a cat.



---

## 1.2 The Water Divider: The XOR Problem

To understand why Deep Learning has become so essential, we need to talk about the **XOR (Exclusive OR) Problem**. In logic, XOR only results in "1" if the inputs are different:

* (0, 0) $\rightarrow$ 0
* (1, 1) $\rightarrow$ 0
* **(1, 0) $\rightarrow$ 1**
* **(0, 1) $\rightarrow$ 1**

### Linear Limitation
Simple Machine Learning models (like the original Perceptron) are **linear**. This means they try to separate the data with a **straight line**. If you plot the XOR points on a graph, you will see that it is impossible to separate the "1" results from the "0" results with a single line. 



### The Deep Learning Solution
Deep Learning solves the XOR problem by adding extra layers with **Activation Functions**. By doing so, the network gains the ability to **"bend" mathematical space**. Instead of a straight line, it creates complex, curved decision boundaries that wrap around the data seamlessly.

---

## 1.3 Neural Networks vs. Ensembles (Random Forest / XGBoost)

Many people ask why we don't just use **Ensemble** models (like Random Forest), which are excellent. The difference is in the type of problem:

1.  **Structured vs. Unstructured Data:**
    * **Ensembles:** They are the kings of tables (Excel/SQL). Great for predicting Churn or credit approval.
    * **Deep Learning:** It is sovereign over unstructured data (Images, Audio, Text).

2.  **Feature Extraction (Features):**
    * In classical ML, a human needs to prepare the data and say what is important.
    * In Deep Learning, the network performs **Automatic Extraction**. It figures out on its own what "weights" are important in the pixels of the image, without you having to define what an eye or an ear is.

# 2. Training Strategies and Evolution

Now that we understand the concept of Deep Learning, how information is created, the basis of models and more. We need to understand **how** training is orchestrated in practice. Learning is not a single event, but a process that can be divided into different phases and methodologies.

---

## 2.1 Classical Training (The 3 Pillars)

As we discussed earlier, the basis of how we teach a machine falls into three broad categories, depending on the presence or absence of a "right answer" (target/label):

* **Supervised Learning (Supervised Learning):** The model learns from labeled examples. It's the "Study with Answer Key". We use it for Regression and Classification.
* **Unsupervised Learning (Unsupervised Learning):** The model searches for hidden patterns and structures in the data on its own. It's "Standard Recognition". We use it for Clustering (Clustering).
* **Reinforcement Learning (Reinforcement Learning):** The model learns through trial and error, receiving rewards or punishments. It is "Learning by Experience".



---

## 2.2 The Lifecycle of a Modern Model

In today's AI landscape, we rarely train a giant model from scratch for every problem. Instead, we follow a "specialization" flow:

### 1. Pre-training (Pre-training)
It is the phase in which the model is trained on a massive, generic dataset to learn the foundations of the world.
* **Example:** A language model reads all of Wikipedia and millions of books just to learn sentence structure and word meaning.
* **Result:** A "base" model that has a lot of general knowledge but no specialties.

### 2. Transfer Learning (Transfer Learning)
It's the act of taking this pre-trained model and "transferring" knowledge from it to a new problem.
* **The Logic:** If the model has already learned to identify shapes and edges in millions of photos, it doesn't need to learn it again to identify cancer cells. He already has the "visual foundation".

### 3. Fine-tuning (Fine Fit)
It's refinement. We took the model that came from Transfer Learning and lightly trained it on a very specific, smaller dataset.
* **The Process:** We adjust the network's final **Weights ($w$)** so that it specializes in the desired task (e.g., detecting fraud on bank-specific credit cards).



---

## 2.3 Advanced Training Concepts

To ensure this training occurs efficiently, we use techniques that control the flow of information:

* **Self-Supervised Learning:** A variation where the model creates its own labels from the data (e.g., hiding a word in a sentence and trying to guess it). It is the basis of GPT models.
* **Active Learning:** The model identifies which data it has the most difficulty understanding and asks a human to specifically label those examples.
* **Online Learning:** The model continues learning and updating its weights while receiving new data in real time, without having to stop for a complete new workout.

---

## 2.4 Technical Summary: When to use what?

| Method | When to use? | Human Analogy |
| :- | :- | :- |
| **Pre-training** | When you want to create a general knowledge base. | Elementary and High School. |
| **Transfer Learning** | When you have little specific data, but there is a similar model ready. | Take advantage of the fact that you know how to play guitar to learn bass. |
| **Fine-tuning** | When you want maximum accuracy on a specific task. | Take a specialization or PhD in a topic. |
| **Reinforcement** | When the goal is to make decisions in sequence (games, robotics). | Learn to ride a bike by falling and getting up. |

---

> **Navigation Warning:** The next step to mastering Neural Networks is to understand what happens "under the hood" during these workouts: calculating **Backpropagation** and choosing the **Optimizer**.