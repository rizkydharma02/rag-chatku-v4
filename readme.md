## Chatku AI LLMs with Retrieval Augmented Generation V4

Aplikasi web Retrieval Augmented Generation menggunakan Python, Streamlit dan LangChain, sehingga Anda dapat mengobrol dengan Dokumen, Situs Web, dan data khusus lainnya.

Jalankan Lokal:

$ git clone <this-repo-url>

$ cd <this-repo-folder>

$ python -m venv venv #opsional

$ venv\Scripts\activate #opsional atau source venv/bin/activate in Linux/Mac

$ pip install -r requirements.txt

$ streamlit run app.py

### .env

Env Menggunakan Groq Cloud Key dan Database Cloud Menggunakan Supabase

DATABASE_URL = "postgresql://postgres.dltxpmhhuwpopnhyrijt:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/namadatabase"
GROQ_API_KEY = ""
JWT_SECRET_KEY=""

Note: Python version 3.11
