# demo3_ai

An experimental AI-powered automation project built with **Python** and **LangChain** to explore integrations like FAISS vector stores, AI chat models, and automation workflows.

## 📌 Features
- AI agent integration using LangChain.
- FAISS vector store for semantic search.
- Modular Python project structure for easy extension.
- Version-controlled with Git & GitHub.

## 🛠 Tech Stack
- **Python 3.10+**
- **LangChain**
- **FAISS** (via `langchain_community.vectorstores`)
- **Git** for version control

## 📂 Project Structure


demo3_ai/
├── main.py # Main entry point
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── ... (other scripts/modules)



## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/devyanik77/demo3_ai.git
cd demo3_ai


python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate


python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate


pip install -r requirements.txt


How It Works?
LangChain orchestrates AI calls.
FAISS indexes and retrieves relevant vectors.
The modular code allows swapping different AI backends or vector DBs.


