

## 🚀 Requirements

Before starting, make sure you have:

* Python 3.8+
* Ollama installed on your system

---

## ⚙️ Installation

### 1. Install Ollama

Download and install Ollama from the official website:
👉 [https://ollama.com](https://ollama.com)

After installation, verify it works:

```bash
ollama --version
```

---

### 2. Run the Model

Start the required model:

```bash
ollama run gemma4:2eb
```

---

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

* On Windows:

```bash
venv\Scripts\activate
```

* On macOS/Linux:

```bash
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the coach with your input file:

```bash
python Coach.py file.txt
```

* `Coach.py` → Main script
* `file.txt` → Your activity log or input data

---

## 📄 Example Input (file.txt)

```
06:30 - Woke up, checked phone for 45 minutes
08:00 - Started working
09:30 - Distracted by YouTube
```

---



