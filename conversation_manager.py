# conversation_manager.py
from collections import deque
from config import CONVERSATION_BUFFER_SIZE


class ConversationManager:
    """대화 관리를 담당하는 클래스"""

    def __init__(self, llm_utils):
        self.llm_utils = llm_utils
        self.conversation_buffer = deque(maxlen=CONVERSATION_BUFFER_SIZE)
        self.conversation_summary = "아직 대화를 시작하지 않았다."

    def add_message(self, speaker: str, message: str):
        """대화 메시지 추가"""
        self.conversation_buffer.append((speaker, message))

    def summarize_conversation(self):
        """현재 대화를 요약"""
        print("DEBUG (Context): 현재 대화 맥락을 요약합니다...")

        if not self.conversation_buffer:
            return

        buffer_str = "\n".join([f"{spk}: {txt}" for spk, txt in self.conversation_buffer])
        prompt = (
            "다음은 최근 대화 기록이야. 이 대화의 현재 주제와 흐름을 한 문장으로 요약해줘.\n\n"
            f"[대화 기록]\n{buffer_str}\n\n[요약]"
        )
        summary = self.llm_utils.get_llm_response(prompt, temperature=0.3, max_tokens=100)
        self.conversation_summary = summary
        print(f"DEBUG (Context): 요약된 현재 대화 맥락 -> {self.conversation_summary}")

    def get_conversation_summary(self) -> str:
        """대화 요약 반환"""
        return self.conversation_summary

    def clear_buffer(self):
        """대화 버퍼 초기화"""
        self.conversation_buffer.clear()
        self.conversation_summary = "아직 대화를 시작하지 않았다."