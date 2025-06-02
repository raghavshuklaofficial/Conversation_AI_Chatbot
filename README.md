# 🧠 **Conversation AI Chatbot**

*An intelligent chatbot built using Natural Language Processing (NLP) and deep learning techniques.*

## 📋 **Overview**

This project implements a conversational AI chatbot capable of understanding user inputs and providing relevant responses. It leverages NLP for intent recognition and a neural network model for response generation. The chatbot can be used via a command-line interface or a GUI.

## 🚀 **Features**

* 💬 Real-time conversation handling
* 🧠 Intent recognition using NLP
* 🤖 Deep learning-based response generation
* 🖥️ GUI-based chatbot interface
* 📂 Modular and extensible codebase

## 🛠️ **Tech Stack**

| **Component**   | **Technology**       |
| --------------- | -------------------- |
| 🐍 Language     | `Python 3.x`         |
| 📚 Libraries    | `TensorFlow`, `TFLearn`, `NumPy`, `NLTK` |
| 🖼️ GUI          | `Tkinter`            |
| 📦 Dependencies | `json`, `pickle`, `os` |

## 📁 **Project Structure**

```plaintext
Conversation_AI_Chatbot/
├── dataset/
│   └── dataset.json               # Training data for the chatbot
├── models/
│   └── chatbot-model.tflearn      # Trained model files
├── data.pickle                    # Preprocessed data for training
├── model_training.py              # Script to train the chatbot model
├── conv_chatbot.py                # Script to run the chatbot
├── main.py                        # GUI interface for the chatbot
└── README.md                      # Project documentation
```

## ⚙️ **Setup & Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/raghavshuklaofficial/Conversation_AI_Chatbot.git
   cd Conversation_AI_Chatbot
   ```

2. **Install dependencies:**

   ```bash
   pip install tensorflow tflearn numpy nltk
   ```

3. **Train the chatbot model:**

   * Ensure the dataset is present in `dataset/dataset.json`.
   * Run the training script:

     ```bash
     python model_training.py
     ```

4. **Run the chatbot:**

   * For command-line interaction:

     ```bash
     python conv_chatbot.py
     ```

   * For GUI-based interaction:

     ```bash
     python main.py
     ```

## 🧪 **Usage**

* **Adding New Intents:**

  * Update the `dataset/dataset.json` file with new intents, patterns, and responses.
  * Re-run `model_training.py` to retrain the model.

* **Interacting with the Chatbot:**

  * Use the GUI or command-line interface to chat with the bot.

---

## 👨‍💻 **Author**

**Raghav Shukla**  
📌 [GitHub Profile](https://github.com/raghavshuklaofficial)

---

## 📄 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
