# npc_agent_autonomous.py
from memory_manager import MemoryManager
from conversation_manager import ConversationManager
from autonomous_planner import AutonomousPlanner
from action_executor import ActionExecutor
from time_manager import time_manager
from config import REFLECTION_THRESHOLD


class AutonomousNpcAgent:
    """자율 행동 기능이 추가된 NPC 에이전트 클래스"""

    def __init__(self, name: str, persona: str, llm_utils):
        self.name = name
        self.persona = persona
        self.llm_utils = llm_utils

        # 기존 컴포넌트들
        self.memory_manager = MemoryManager(llm_utils, name)
        self.memory_manager.set_persona_description(persona)
        self.conversation_manager = ConversationManager(llm_utils)

        # 새로운 자율 행동 컴포넌트들
        self.planner = AutonomousPlanner(self, llm_utils)
        self.executor = ActionExecutor(self, llm_utils)

        # 상태 정보
        self.current_situation = "일상 생활 중"
        self.current_emotion = "평온함"
        self.current_goal = "하루 일과를 보내며 필요시 플레이어와 상호작용하기"
        self.current_location = "대학교:중앙광장"

        # 자율 행동 관련 상태
        self.is_autonomous_mode = True
        self.last_autonomous_update = None
        self.autonomous_update_interval = 60  # 60초마다 업데이트

        # ⭐ 추가: Unity로부터 월드 정보를 받아 계획 수립이 가능한지 여부
        self.is_ready_for_planning = False

        # 리플렉션 관련
        self.reflection_importance_sum = 0
        self.reflection_threshold = REFLECTION_THRESHOLD

        # 플레이어 상호작용 상태
        self.is_interacting_with_player = False
        self.interaction_start_time = None

        # 초기 기억 설정
        self._initialize_memories()

        # 시간 이벤트 콜백 등록
        time_manager.register_callback('hour', self._on_hour_change)
        time_manager.register_callback('new_day', self._on_new_day)

    def _initialize_memories(self):
        """초기 기억 설정"""
        self.memory_manager.add_memory('event', f"나의 이름은 '{self.name}'이다.", 10)
        self.memory_manager.add_memory('event', f"나의 성격 및 설정: '{self.persona}'", 10)
        self.memory_manager.add_memory('thought', f"[목표] 나의 현재 목표는 '{self.current_goal}'이다.", 9)

    def update_emotion(self, new_emotion: str):
        """감정 상태 업데이트"""
        old_emotion = self.current_emotion
        self.current_emotion = new_emotion

        print(f"[AutonomousNPC] {self.name}: 감정 변화 '{old_emotion}' → '{new_emotion}'")


    def _on_hour_change(self, current_time):
        """시간 변화 시 호출되는 콜백"""
        print(f"[AutonomousNPC] {self.name}: 시간 변화 감지 - {current_time.strftime('%H:%M')}")

        # 새로운 계획이 필요한지 확인
        if self.planner.should_replan(current_time):
            self.planner.create_new_daily_plan(current_time)

        # 다음 행동 결정
        if not self.is_interacting_with_player:
            self.executor.determine_next_action(current_time, self.planner)

    def _on_new_day(self, current_time):
        """새로운 날 시작 시 호출되는 콜백"""
        print(f"[AutonomousNPC] {self.name}: 새로운 날 시작 - {current_time.strftime('%Y-%m-%d')}")

        # 새로운 날의 계획 수립
        self.planner.create_new_daily_plan(current_time)

        # 감정 상태 초기화
        self.current_emotion = "상쾌함"

        # 어제 하루를 돌아보는 생각 추가
        self.memory_manager.add_memory(
            'thought',
            f"새로운 날이 시작되었다. 오늘도 열심히 지내야겠다.",
            importance=6
        )

    def autonomous_update(self):
        """자율 행동 업데이트 (주기적으로 호출)"""


        """자율 행동 업데이트 (주기적으로 호출)"""
        current_time = time_manager.get_current_time()

        # 업데이트 간격 체크
        if (self.last_autonomous_update and
                (current_time - self.last_autonomous_update).seconds < self.autonomous_update_interval):
            return

        print(f"[AutonomousNPC] {self.name} 자율 행동 업데이트")

        # 플레이어와 상호작용 중이 아닐 때만 자율 행동
        if not self.is_interacting_with_player:

            # 1. 현재 계획 확인 및 필요시 새 계획 생성
            if self.planner.should_replan(current_time):
                self.planner.create_new_daily_plan(current_time)

            # 2. 현재 행동 상태 확인 및 다음 행동 결정
            action_changed = self.executor.determine_next_action(current_time, self.planner)

            if action_changed:
                # 3. 새로운 행동에 대한 생각 기록
                status = self.executor.get_current_status()
                self.memory_manager.add_memory(
                    'thought',
                    f"지금 {status['location']}에서 {status['action']}를 하고 있다.",
                    importance=4
                )

        self.last_autonomous_update = current_time

    def respond_to_player(self, player_input: str, player_location: str = None) -> str:
        """플레이어 입력에 대한 응답 생성 (기존 메서드 확장)"""
        print(f"[AutonomousNPC] {self.name}: 플레이어 상호작용 시작")

        # 플레이어 상호작용 모드로 전환
        self.is_interacting_with_player = True
        self.interaction_start_time = time_manager.get_current_time()

        # 현재 상황 정보 수집
        current_status = self.executor.get_current_status()
        current_time = time_manager.get_current_time()

        # 상호작용 처리
        if player_location:
            self.executor.handle_player_interaction(player_location, "chat")

        # 대화 기록 추가
        self.conversation_manager.add_message("Player", player_input)

        # 관련 기억 및 지식 검색
        relevant_memories = self.memory_manager.retrieve_memories(player_input)
        relevant_knowledge = self.memory_manager.retrieve_knowledge(player_input)

        # 컨텍스트 생성 (현재 상황 포함)
        memory_context = "\n".join([f"- {m.description}" for m in relevant_memories])
        knowledge_context = "\n".join(relevant_knowledge)

        # 현재 상황 컨텍스트
        situation_context = f"""
        현재 시간: {current_time.strftime('%H:%M')}
        현재 위치: {current_status['location']}
        현재 하던 일: {current_status['description']}
        현재 감정: {self.current_emotion}
        """

        # 응답 생성
        response = self._generate_contextual_response(
            player_input, memory_context, knowledge_context, situation_context
        )

        # 대화 종료 후 처리
        self._handle_interaction_end(player_input, response)

        return response

    def _generate_contextual_response(self, player_input: str, memory_context: str,
                                    knowledge_context: str, situation_context: str) -> str:
        """상황을 고려한 응답 생성 (메타 정보 포함)"""

        # 🔸 회상 메모리에서 메타 정보 추출
        meta_lines = []
        for m in self.memory_manager.retrieve_memories(player_input, top_k=5):
            if hasattr(m, "strategy") and m.strategy:
                meta_lines.append(f"AI 전략: {m.strategy}")
            if hasattr(m, "emotion") and m.emotion:
                meta_lines.append(f"사용자 감정: {m.emotion}")
            if hasattr(m, "personality") and m.personality:
                meta_lines.append(f"사용자 성격: {m.personality}")
        meta_context = "\n".join(meta_lines)

        # 🔸 최종 프롬프트 구성
        response_prompt = f"""
        너는 '{self.name}'({self.persona})이야

        ### 회상된 메타 정보 ###
        {meta_context}

        ### 현재 상황 ###
        {situation_context}

        ### 현재 대화의 핵심 흐름 ###
        {self.conversation_manager.get_conversation_summary()}

        ### 너가 알고 있는 사실 (지식 베이스) ###
        {knowledge_context}

        ### 너의 장기 기억 (과거 사건 및 생각) ###
        {memory_context}

        ### 방금 일어난 일 ###
        플레이어가 방금 너에게 이렇게 말했어: "{player_input}"

        ### 지시문 ###
        위의 모든 정보(특히 '현재 상황', '메타 정보', '대화 흐름')를 고려하여,
        플레이어에게 할 가장 자연스러운 다음 응답을 한 문장으로 생성해줘.
        현재 하던 일이나 감정 상태를 자연스럽게 반영해서 대답해.
        """

        return self.llm_utils.get_llm_response(response_prompt)
    
    
    def _handle_interaction_end(self, player_input: str, response: str):
        """상호작용 종료 후 처리"""
        # 기억 추가
        self.memory_manager.add_memory('event', f"플레이어가 나에게 '{player_input}'라고 말했다.", 6)
        self.memory_manager.add_memory('event', f"나는 플레이어에게 '{response}'라고 대답했다.", 6)

        # 대화 기록 및 요약
        self.conversation_manager.add_message(self.name, response)
        self.conversation_manager.summarize_conversation()

        # 지식 학습
        interaction = f"Player: {player_input}\n{self.name}: {response}"
        learned_concepts = self.memory_manager.learn_from_interaction(interaction)

        # 새로 학습한 지식에 대한 메모리 추가
        if learned_concepts and isinstance(learned_concepts, dict):
            for concept, desc in learned_concepts.items():
                self.memory_manager.add_memory('thought',
                                               f"[지식 습득] '{concept}'은(는) '{desc}'라는 것을 알게 되었다.", 7)

        # 플레이어와의 상호작용 경험을 바탕으로 감정 업데이트
        self._update_emotion_from_interaction(player_input, response)

    def _update_emotion_from_interaction(self, player_input: str, response: str):
        """상호작용을 바탕으로 감정 업데이트"""
        positive_words = ["고마워", "도움", "좋아", "재미있", "기뻐"]
        negative_words = ["싫어", "화나", "슬퍼", "힘들어", "스트레스"]

        if any(word in player_input for word in positive_words):
            self.current_emotion = "기쁨"
        elif any(word in player_input for word in negative_words):
            self.current_emotion = "동정"
        else:
            self.current_emotion = "호기심"

    def end_player_interaction(self):
        """플레이어 상호작용 종료"""
        print(f"[AutonomousNPC] {self.name}: 플레이어 상호작용 종료")

        self.is_interacting_with_player = False
        self.interaction_start_time = None

        # 이전 활동으로 복귀하거나 새로운 활동 결정
        current_time = time_manager.get_current_time()
        self.executor.determine_next_action(current_time, self.planner)

    def get_status_for_unity(self):
        """Unity에 보낼 상태 정보 생성"""
        current_status = self.executor.get_current_status()

        return {
            "npc_id": self.name,
            "name": self.name,
            "current_action": current_status['action'],
            "description": current_status['description'],
            "emoji": current_status['emoji'],
            "location": current_status['location'],
            "emotion": self.current_emotion,
            "is_busy": self.is_interacting_with_player,
            "movement_command": self.executor.get_unity_movement_command(),
            "interaction_available": not self.is_interacting_with_player,
            "current_thought": self._get_current_thought()
        }

    def _get_current_thought(self):
        """현재 생각 생성"""
        status = self.executor.get_current_status()

        if self.is_interacting_with_player:
            return "플레이어와 대화 중이에요"
        elif status['action']:
            return f"{status['description']} 중이에요"
        else:
            return "뭘 할지 생각 중이에요"

    def get_debug_info(self):
        """디버그 정보 반환"""
        current_time = time_manager.get_current_time()

        return {
            "name": self.name,
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": self.current_emotion,
            "location": self.current_location,
            "autonomous_mode": self.is_autonomous_mode,
            "interacting_with_player": self.is_interacting_with_player,
            "current_action": self.executor.get_current_status(),
            "daily_schedule": self.planner.get_schedule_summary(),
            "memory_count": len(self.memory_manager.seq_event) + len(self.memory_manager.seq_thought),
            "knowledge_count": len(self.memory_manager.knowledge_base)
        }