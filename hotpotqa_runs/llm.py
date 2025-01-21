import os
import sys
import time

import google.generativeai as genai
from typing import Union, Literal

from google.api_core.exceptions import ResourceExhausted
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.schema import (
    HumanMessage
)

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'
    
    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content

class GeminiLLM:
    def __init__(self, *args, **kwargs):
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")

    def __call__(self, prompt: str, stop=["\n"]):
        max_retries = 10
        retry_count = 0
        backoff_factor = 2  # Exponential growth factor for the delay
        delay = 1  # Initial delay in seconds

        while retry_count < max_retries:
            try:
                cur_try = 0
                while cur_try < 6:
                    # breakpoint()
                    response = self.gemini.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=cur_try * 0.2,
                            max_output_tokens=100,
                            top_p=1,
                            stop_sequences=stop
                        )
                    )
                    # text = response["choices"][0]["text"]
                    text = response.text
                    text = text.lstrip().removeprefix('Action: ').strip()
                    # Dumb way to check text length
                    if len(text.strip()) >= 5:
                        return text
                    cur_try += 1
                return ""
            except Exception as e:
                print(f"Request failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retry_count += 1
                delay *= backoff_factor

        print('Max retries reached')
        sys.exit(1)