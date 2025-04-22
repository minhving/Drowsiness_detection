from openai import OpenAI
import os
from dotenv import load_dotenv

class OpenAi:
    def __init__(self):
        self.client = None

    def initialize(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
    def response(self):
        with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input="Please wait for the ChatGPT",
                instructions="Speak in a cheerful and positive tone.",
        ) as response:
            response.stream_to_file("output2.mp3")

    def response_to_require(self, filename):
        with open(filename, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt="There are only two options: Yes or No. Please respond with either Yes or No."
            )
            text = transcription.text
            print("📝 Transcribed:", text)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "If the user says yes, explain how to stop a car. If no, explain why drowsy driving is dangerous. Answer should be in 3 short sentences for driver to listen while driving but can fully understand it."},
                {"role": "user", "content": text}
            ]
        )

        reply = response.choices[0].message.content.strip()
        print("🤖 GPT:", reply)

        with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input= reply,
                instructions="Speak in positive tone.",
        ) as response:
            response.stream_to_file("output1.mp3")
        print("Sucess translate")
