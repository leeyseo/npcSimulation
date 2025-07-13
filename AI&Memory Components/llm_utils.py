# llm_utils.py
import openai
import time
from CoreSystemFiles.config import EMBEDDING_MODEL, CHAT_MODEL


class LLM_Utils:
    """OpenAI API를 활용한 LLM 유틸리티 클래스"""

    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key or "sk-" not in self.api_key:
            self.api_key = None
            print("### 경고: OpenAI API 키가 유효하지 않습니다. ###")
        else:
            openai.api_key = self.api_key

    def get_llm_response(self, prompt: str, temperature=0.7, max_tokens=150, is_json=False) -> str:
        """LLM으로부터 응답을 받아오는 메서드"""
        if not self.api_key:
            return f"(API 키 없음) 임시 응답."

        try:
            time.sleep(0.5)  # API 호출 간격 조절

            messages = [
                {"role": "system",
                 "content": "You are a thoughtful AI character. Please follow the user's instructions precisely."}
            ]

            if is_json:
                messages[0]["content"] += " You must respond in the requested JSON format."

            messages.append({"role": "user", "content": prompt})

            response_format = {"type": "json_object"} if is_json else None

            completion = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            return f"죄송해요, 오류가 발생했어요: {e}"

    def get_embedding(self, text: str) -> list[float]:
        """텍스트의 임베딩을 생성하는 메서드"""
        if not self.api_key:
            return [0.0] * 1536

        try:
            if not text or not text.strip():
                return [0.0] * 1536

            text = text.replace("\n", " ")
            response = openai.embeddings.create(
                input=[text],
                model=EMBEDDING_MODEL
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return [0.0] * 1536