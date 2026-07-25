import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("azure_endpoint"),
    api_key=os.getenv("openai_key_azure"),
)

MODEL = "gpt-5.1"


def ask_llm(system_prompt: str, user_prompt: str) -> str:

    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role": "system",
                "content": system_prompt,
            },

            {
                "role": "user",
                "content": user_prompt,
            }

        ],

        temperature=0

    )

    return response.choices[0].message.content.strip()