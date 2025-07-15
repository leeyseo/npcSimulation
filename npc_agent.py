# npc_agent.py
from memory_manager import MemoryManager
from conversation_manager import ConversationManager
from config import REFLECTION_THRESHOLD


class NpcAgent:
    """NPC 에이전트 메인 클래스"""

    def __init__(self, name: str, persona: str, llm_utils):
        self.name = name
        self.persona = persona
        self.llm_utils = llm_utils

        # 컴포넌트 초기화
        self.memory_manager = MemoryManager(llm_utils, name)
        self.memory_manager.set_persona_description(persona)
        self.conversation_manager = ConversationManager(llm_utils)

        # 상태 정보
        self.current_situation = "플레이어와 마주보고 대화하고 있다."
        self.current_emotion = "평온함"
        self.current_goal = "플레이어와 친해지고, 나의 고민에 대한 조언을 얻고 싶다."

        # 리플렉션 관련
        self.reflection_importance_sum = 0
        self.reflection_threshold = REFLECTION_THRESHOLD

        # 초기 기억 설정
        self._initialize_memories()

    def _initialize_memories(self):
        """초기 기억 설정"""
        self.memory_manager.add_memory('event', f"나의 이름은 '{self.name}'이다.", 10)
        self.memory_manager.add_memory('event', f"나의 성격 및 설정: '{self.persona}'", 10)
        self.memory_manager.add_memory('thought', f"[목표] 나의 현재 목표는 '{self.current_goal}'이다.", 9)

    def respond_to_player(self, player_input: str) -> str:
        """플레이어 입력에 대한 응답 생성"""
        # 대화 기록 추가
        self.conversation_manager.add_message("Player", player_input)

        # 관련 기억 및 지식 검색
        relevant_memories = self.memory_manager.retrieve_memories(player_input)
        relevant_knowledge = self.memory_manager.retrieve_knowledge(player_input)

        # ▶️ 메타 정보 추출
        meta_lines = []
        for m in relevant_memories:
            if hasattr(m, "strategy") and m.strategy:
                meta_lines.append(f"AI 전략: {m.strategy}")
            if hasattr(m, "emotion") and m.emotion:
                meta_lines.append(f"사용자 감정: {m.emotion}")
            if hasattr(m, "personality") and m.personality:
                meta_lines.append(f"사용자 성격: {m.personality}")
        meta_context = "\n".join(meta_lines)

        # 컨텍스트 생성
        memory_context = "\n".join([f"- {m.description}" for m in relevant_memories])
        knowledge_context = "\n".join(relevant_knowledge)

        # 응답 생성 (meta_context 추가)
        response = self._generate_response(player_input, memory_context, knowledge_context, meta_context)

        # 기억 추가
        self.memory_manager.add_memory('event', f"플레이어가 나에게 '{player_input}'라고 말했다.")
        self.memory_manager.add_memory('event', f"나는 플레이어에게 '{response}'라고 대답했다.")

        # 대화 기록 및 요약
        self.conversation_manager.add_message(self.name, response)
        self.conversation_manager.summarize_conversation()

        # 지식 학습
        interaction = f"Player: {player_input}\n{self.name}: {response}"
        learned_concepts = self.memory_manager.learn_from_interaction(interaction)

        if learned_concepts and isinstance(learned_concepts, dict):
            for concept, desc in learned_concepts.items():
                self.memory_manager.add_memory('thought',
                    f"[지식 습득] '{concept}'은(는) '{desc}'라는 것을 알게 되었다.", 7)

        return response

    def _generate_response(self, player_input: str,
                        memory_context: str,
                        knowledge_context: str,
                        meta_context: str = "") -> str:
        """응답 생성 (메타 정보 포함)"""
        response_prompt = f"""
        너는 '{self.name}'({self.persona})이야

        ### 사용자 메타 정보 ###
        {meta_context}

        ### 현재 대화의 핵심 흐름 ###
        {self.conversation_manager.get_conversation_summary()}

        ### 너가 알고 있는 사실 (지식 베이스) ###
        {knowledge_context}

        ### 너의 장기 기억 (과거 사건 및 생각) ###
        {memory_context}

        ### 방금 일어난 일 ###
        플레이어가 방금 너에게 이렇게 말했어: "{player_input}"

        ### 지시문 ###
        위의 모든 정보(특히 '사용자 메타 정보'와 '현재 대화의 핵심 흐름')를 가장 중요하게 고려하여,
        플레이어에게 할 가장 자연스러운 다음 응답을 한 문장으로 생성해줘.
        """
        
        return self.llm_utils.get_llm_response(response_prompt)

    def perform_reflection(self):
        """리플렉션 수행 (현재는 빈 구현)"""
        # 추후 구현 예정
        pass

    def _should_reflect(self) -> bool:
        """리플렉션이 필요한지 판단"""
        return self.reflection_importance_sum >= self.reflection_threshold

    def update_emotion(self, new_emotion: str):
        """감정 상태 업데이트"""
        self.current_emotion = new_emotion
        self.memory_manager.add_memory('thought', f"내 감정이 '{new_emotion}'으로 바뀌었다.", 6)

    def update_goal(self, new_goal: str):
        """목표 업데이트"""
        self.current_goal = new_goal
        self.memory_manager.add_memory('thought', f"내 목표가 '{new_goal}'으로 바뀌었다.", 8)

    def get_status(self) -> dict:
        """현재 상태 반환"""
        return {
            "name": self.name,
            "persona": self.persona,
            "current_emotion": self.current_emotion,
            "current_goal": self.current_goal,
            "current_situation": self.current_situation,
            "total_memories": len(self.memory_manager.seq_event) + len(self.memory_manager.seq_thought),
            "knowledge_count": len(self.memory_manager.knowledge_base),
            "conversation_summary": self.conversation_manager.get_conversation_summary()
        }