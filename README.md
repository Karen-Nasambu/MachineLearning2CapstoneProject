# MachineLearning2CapstoneProject

# 🕵️‍♂️ Unsupervised Anti-Money Laundering (AML) Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![VS Code](https://img.shields.io/badge/Editor-VS%20Code-007ACC)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📖 Project Description
This project implements an advanced **Unsupervised Anomaly Detection system** designed to identify potential money laundering activities within cryptocurrency transaction networks.

Unlike traditional supervised models that rely on historical labels (which are often scarce or outdated), this solution utilizes **Graph Neural Networks (GNNs)** to learn the inherent structure of legitimate financial behavior. By treating the financial network as a graph—where accounts are nodes and transactions are edges—the model flags "structural anomalies" that deviate from the norm, effectively catching suspicious actors without prior knowledge of their specific identity.

**Dataset Source:**
The model is trained on the **Elliptic Data Set**, a sub-graph of the Bitcoin blockchain containing over 200,000 transactions.
* **Download Link:** [Kaggle - Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

---

## 🚩 Problem Statement: What are we solving?
**The Challenge:**
Financial crime is becoming increasingly sophisticated. Money launderers use complex, layered chains of transfers ("smurfing" or "layering") to obscure the illicit origin of funds.
1.  **Rule-Based Failures:** Traditional "If/Then" rules (e.g., "flag transactions over $10k") generate too many false positives and are easily bypassed by criminals splitting amounts.
2.  **Lack of Labeled Data:** In the real world, banks do not have a pre-labeled list of all criminals. Most financial data is unlabeled, making standard supervised learning impossible to deploy effectively for new, unknown threats.

**The Goal:**
To build a system that can answer: *"Is this transaction suspicious based on its relationship to the rest of the network, even if we've never seen this specific crime pattern before?"*

---

## 🛠️ Tools Used
This project will be  built using a modern Data Science and Machine Learning stack within **Visual Studio Code**.

* **Language:** [Python](https://www.python.org/) (Data manipulation and modeling)
* **Deep Learning Framework:** [PyTorch](https://pytorch.org/) & [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/) (Implementing Graph Convolutional Networks)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) (Preprocessing 200k+ transaction rows)
* **Visualization:** [Matplotlib](https://matplotlib.org/) & [NetworkX](https://networkx.org/) (Visualizing graph structures and loss curves)
* **Environment Management:** Anaconda / Python venv
* **IDE:** Visual Studio Code (VS Code)

---

## 💡 Insights & Solutions
### The "Aha!" Moment
During the analysis, we discovered that illicit transactions (money laundering) form distinct **topological patterns** compared to licit ones. While a normal user sends money directly to an exchange or a merchant, launderers create dense, cyclical clusters to hide the money trail.

### The Solution Offered
We developed a **Graph Autoencoder (GAE)**.
1.  **Mechanism:** The model compresses the transaction network into a low-dimensional code and attempts to reconstruct it.
2.  **Detection Logic:** The model learns to reconstruct "normal" transactions perfectly. When it encounters a laundering ring, the reconstruction fails (high error rate) because the pattern is mathematically "weird."
3.  **Result:** An automated **"Risk Score"** for every transaction.
    * *Low Score:* Safe / Normal Business.
    * *High Score:* Potential Laundering / Requires Manual Review.

---

## 💼 Business Impact
How does this benefit a Financial Institution or Compliance Team?

1.  **Detection of "Zero-Day" Crime:** Unlike rule-based systems that only catch *known* methods, this unsupervised approach detects *new* anomalies, protecting the bank from emerging threats.
2.  **Operational Efficiency:** By ranking transactions by "Risk Score," compliance officers can focus their limited time on the top 1% most suspicious cases rather than reviewing thousands of false alarms.
3.  **Regulatory Compliance:** Reduces the risk of massive fines (AML non-compliance) by demonstrating a state-of-the-art, proactive monitoring capability.

---

## 🚀 Deployment
To make these insights accessible to stakeholders, the project deployment strategy is as follows:

* **Documentation & Reporting:** The project analysis, interactive notebooks, and final report are hosted via **GitHub Pages**. This serves as the central knowledge hub for the technical implementation and business insights.
* *(Future Integration):* The core model is designed to be wrapped in a REST API (using FastAPI) which can then be connected to a frontend dashboard (built with tools like **Lovable** or **Streamlit**) for real-time transaction scoring by bank analysts.

---

Here is the comprehensive `README.md` file for your project. You can copy this directly into your GitHub repository or project documentation.

It is written to sound professional, technically impressive, and deeply impactful.

---

# 📸 The Talking Lens: AI Scene Describer for the Visually Impaired

### *Giving sight to the blind through the power of Multimodal Deep Learning.*

---

## 📖 Table of Contents

1. [The Problem Statement](https://www.google.com/search?q=%23-the-problem-statement)
2. [The Solution & Insights](https://www.google.com/search?q=%23-the-solution--insights)
3. [Social Impact](https://www.google.com/search?q=%23-social-impact)
4. [Tech Stack & Tools](https://www.google.com/search?q=%23-tech-stack--tools)
5. [The Dataset](https://www.google.com/search?q=%23-the-dataset)
6. [Project Architecture](https://www.google.com/search?q=%23-project-architecture)
7. [Installation & Usage](https://www.google.com/search?q=%23-installation--usage)
8. [Deployment Strategy](https://www.google.com/search?q=%23-deployment-strategy)

---

## 🚨 The Problem Statement

**"What is in front of me?"**
For the 2.2 billion people with vision impairment globally, this simple question often requires human assistance. While existing tools can detect objects (e.g., "Table," "Dog"), they lack **contextual awareness**.

Knowing there is a "car" is not enough. A blind pedestrian needs to know: *"A red car is speeding down the wet road."*
Knowing there is a "bottle" is not enough. They need to know: *"A clear plastic bottle sitting on the edge of the table."*

Current solutions fail to bridge the gap between **Computer Vision** (Seeing) and **Natural Language** (Describing).

---

## 💡 The Solution & Insights

**The "Aha!" Moment:**
We discovered that we cannot treat Vision and Language as separate problems. By mapping images and text into the **same mathematical space (Vector Embeddings)**, we can teach an AI to "translate" pixels into sentences.

**Our Solution:**
**The Talking Lens** is an end-to-end Generative AI application that:

1. **See:** Takes a photo of the user's surroundings.
2. **Think:** Uses a Convolutional Neural Network (CNN) to extract visual features.
3. **Describe:** Uses a Recurrent Neural Network (LSTM) to generate a full, human-like sentence.
4. **Speak:** Instantly converts that text into spoken audio using Google Text-to-Speech (gTTS).

---

## ❤️ Social Impact

**How does the community benefit?**

* **Independence:** Users can navigate new environments without a human guide.
* **Privacy:** Users can read personal mail or prescription labels without asking strangers for help.
* **Safety:** Real-time description of hazards (e.g., "A construction hole in the sidewalk").

---

## 🛠 Tech Stack & Tools

| Component | Tool / Library | Purpose |
| --- | --- | --- |
| **Language** | Python 3.9 | Core programming logic. |
| **Deep Learning** | **PyTorch** | Building the Neural Networks. |
| **Vision (The Eye)** | **ResNet50** | Pre-trained CNN to extract features from images. |
| **Language (The Brain)** | **LSTM** | Long Short-Term Memory network to generate sentences. |
| **Audio** | **gTTS** | Google Text-to-Speech API for audio output. |
| **Data Handling** | Pandas & NumPy | Loading CSVs and matrix operations. |
| **Deployment** | Streamlit | Creating the user-friendly web interface. |

---

## 📂 The Dataset

We use the **Flickr8k Dataset**, a gold-standard benchmark for image captioning.

* **Source:** [Kaggle - Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* **Size:** ~1 GB
* **Structure:**
* `Images/`: A folder containing 8,092 JPEG images.
* `captions.txt`: A CSV file linking images to descriptions.
* *Format:* `1000268201.jpg, A child in a pink dress is climbing up a set of stairs.`





---

## 🏗 Project Architecture

1. **Encoder (ResNet50):** We remove the last classification layer of ResNet50. Instead of predicting "Dog," it outputs a vector of 2,048 numbers representing the *content* of the image.
2. **Decoder (LSTM):** This network takes the image vector as the initial state. It predicts the caption word-by-word (e.g., "A" -> "Brown" -> "Dog" -> "Running").
3. **Integration:** The final text is passed to the TTS engine to generate an MP3 file played back to the user.

---

## 💻 Installation & Usage

### 1. Clone the Repository


### 2. Install Dependencies

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib gTTS streamlit

```

### 3. Download Data

Download the **Flickr8k** dataset from Kaggle and place the `Images` folder and `captions.txt` inside a `data/` directory in this project.

### 4. Run the Training (Optional)

If you want to train the model from scratch:

```bash
python train_model.py

```

### 5. Run the Application

To launch the "Lovable" web interface:

```bash
streamlit run app.py

```

---

## 🚀 Deployment Strategy

To make this project accessible to the real world, we deploy it using **Streamlit Cloud** or **Lovable.dev**.

**The User Journey:**

1. **Open App:** User opens the web link on their phone.
2. **Snap:** They tap the large "Camera" button.
3. **Listen:** Within 2 seconds, the phone speaks: *"I see a group of friends sitting around a campfire at night."*

---

### **Future Improvements**

* **Attention Mechanism:** implementing visual attention so the model can point to *where* the object is.
* **Multilingual Support:** describing scenes in Spanish, French, and Swahili.

---
## 🏆 Competitive Analysis: How We Stand Out

There are existing solutions for the visually impaired, but they often lack the **contextual storytelling** that "The Talking Lens" provides. Most current tools tell you *what* is there, but not *what is happening*.

### 📱 The "Other" Apps (Current Solutions)

1.  **Google Lens / Bixby Vision:**
    * *What they do:* Excellent at identifying specific objects ("Chair", "Coke Bottle") or reading text.
    * *The Gap:* They are encyclopedic, not descriptive. If you point it at a family dinner, it might list tags like "Table," "Food," "Person." It fails to say: *"A family eating dinner together at a round table."*
2.  **Be My Eyes:**
    * *What they do:* Connects blind users with real human volunteers via video call.
    * *The Gap:* **Privacy & Speed.** Users often don't want a stranger seeing their messy bedroom or personal documents. It also requires waiting for a volunteer to pick up.
3.  **Microsoft Seeing AI:**
    * *What they do:* A powerful multi-tool (reads currency, recognizes faces).
    * *The Gap:* **Complexity.** The user must constantly switch "Channels" (Scene mode vs. Document mode vs. Product mode). Our solution is a single-button "Snap & Describe" interface.

### ✨ The "Talking Lens" Advantage

Our project uses **Generative Multimodal Deep Learning**, which allows it to generate human-like sentences rather than just listing keywords.

| Feature | 🤖 Google Lens | 👥 Be My Eyes | 📸 The Talking Lens (Ours) |
| :--- | :--- | :--- | :--- |
| **Output Type** | List of Tags ("Dog", "Grass") | Human Conversation | **Full Descriptive Sentence** |
| **Privacy** | High | Low (Human sees video) | **High (100% AI-based)** |
| **Speed** | Fast | Slow (Wait time) | **Instant** |
| **Context** | ❌ (Just Objects) | ✅ (Human Intelligence) | **✅ (AI Scene Understanding)** |
| **Availability**| Always | Depends on Volunteers | **24/7 Always On** |

> **Key Differentiator:** While Google Lens helps you *shop*, and Be My Eyes helps you *navigate*, The Talking Lens helps you **understand your environment** through natural language storytelling.

*Project created by [KAREN NASAMBU] for the Data Science Capstone.*

