import mlflow
from groq import Groq
import os
import pandas as pd


# you must set the OPENAI_API_KEY environment variable
os.environ["GROQ_API_KEY"] =   "gsk_.."
mlflow.set_tracking_uri("mongodb://localhost:27017/mlflow")

# Create or get the experiment
try:
    experiment = mlflow.get_experiment_by_name('llm_tracking_test')
    experiment_id = experiment.experiment_id
except mlflow.exceptions.MlflowException:
    # Experiment doesn't exist, create it
    experiment_id = mlflow.create_experiment('llm_tracking_test')

mlflow.set_experiment(experiment_id=experiment_id)
try:
    prompt = mlflow.genai.load_prompt("prompts:/llm_tracking_test_prompt@production")
except mlflow.exceptions.MlflowException:
    prompt = None

if not prompt:
    prompt = mlflow.genai.register_prompt(name="llm_tracking_test_prompt",template="Explain the importance of {{topic}}.")
    mlflow.set_prompt_alias("llm_tracking_test_prompt", alias="production", version=prompt.version)

print(f"Created prompt '{prompt.name}' (version {prompt.version})")

content = prompt.format(topic="meditation")

mlflow.groq.autolog()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content":content,
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)