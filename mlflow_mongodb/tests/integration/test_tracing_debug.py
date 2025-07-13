import mlflow
from groq import Groq
import os
import pandas as pd

# you must set the OPENAI_API_KEY environment variable
os.environ["GROQ_API_KEY"] =   "GROQ_API_KEY"

mlflow.set_tracking_uri("mongodb://localhost:27017/mlflow")
# set the experiment id
mlflow.set_experiment(experiment_id="1752352026927959")


mlflow.groq.autolog()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)