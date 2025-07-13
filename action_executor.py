# action_executor.py
import random


class ActionExecutor:
    """NPC의 행동 실행을 담당하는 클래스 (논문의 execute 모듈 구현)"""

    def __init__(self, npc_agent, llm_utils):
        self.npc = npc_agent
        self.llm_utils = llm_utils

        # 행동 상태
        self.current_action = None
        self.action_start_time = None
        self.action_duration = 0
        self.target_location = None
        self.action_description = ""
        self.action_emoji = "🤔"

        # 위치 매핑 (활동 -> 장소)
        self.activity_locations = {
            "잠자기": "집:침실",
            "기상": "집:침실",
            "아침 루틴": "집:화장실",
            "아침식사": "집:부엌",
            "점심식사": "카페:식당",
            "저녁식사": "집:부엌",
            "공부": "도서관:열람실",
            "과제": "도서관:열람실",
            "수업": "대학교:강의실",
            "휴식": "카페:휴게실",
            "개인시간": "집:거실",
            "취미활동": "집:거실",
            "운동": "체육관:운동실",
            "쇼핑": "상점:매장",
            "산책": "공원:산책로"
        }

    def determine_next_action(self, current_time, planner):
        """다음 행동 결정 (논문의 _determine_action 구현)"""
        print(f"[ActionExecutor] {self.npc.name}의 다음 행동 결정 중...")

        # 1. 현재 활동이 끝났는지 확인
        if self._is_current_action_finished(current_time):
            # 2. 플래너에서 현재 시간의 활동 가져오기
            activity, duration = planner.get_current_activity(current_time)

            # 3. 활동을 구체적인 행동으로 분해
            detailed_action = self._decompose_activity(activity, duration)

            # 4. 행동에 맞는 위치 결정
            target_location = self._determine_location(detailed_action)

            # 5. 행동 설명과 이모지 생성
            description, emoji = self._generate_action_description(detailed_action, target_location)

            # 6. 새로운 행동 설정
            self._set_new_action(
                action=detailed_action,
                location=target_location,
                description=description,
                emoji=emoji,
                duration=duration,
                start_time=current_time
            )

            return True  # 새로운 행동 설정됨

        return False  # 기존 행동 계속

    def _is_current_action_finished(self, current_time):
        """현재 행동이 끝났는지 확인"""
        if not self.current_action or not self.action_start_time:
            return True

        elapsed_minutes = (current_time - self.action_start_time).total_seconds() / 60
        return elapsed_minutes >= self.action_duration

    def _decompose_activity(self, activity, duration):
        """활동을 구체적인 행동으로 분해 (논문의 task decomposition)"""
        # 간단한 분해 로직 (나중에 LLM으로 확장 가능)
        if "공부" in activity or "과제" in activity:
            actions = ["자료 찾기", "읽기", "정리하기", "문제 풀기"]
            return random.choice(actions)
        elif "휴식" in activity:
            actions = ["음악 듣기", "폰 보기", "멍때리기", "간식 먹기"]
            return random.choice(actions)
        elif "식사" in activity:
            actions = ["메뉴 고르기", "주문하기", "식사하기", "정리하기"]
            return random.choice(actions)
        else:
            return activity

    def _determine_location(self, action):
        """행동에 맞는 위치 결정"""
        for activity_key, location in self.activity_locations.items():
            if activity_key in action:
                return location

        # 기본 위치
        return "대학교:중앙광장"

    def _generate_action_description(self, action, location):
        """행동 설명과 이모지 생성"""
        prompt = f"""
        {self.npc.name}({self.npc.persona})가 {location}에서 "{action}"를 하고 있습니다.

        1. 이 상황을 자연스럽게 설명하는 한 문장을 만들어주세요.
        2. 이 행동을 나타내는 적절한 이모지 하나를 골라주세요.

        형식:
        설명: [행동 설명]
        이모지: [이모지]

        예시:
        설명: 도서관에서 과제 자료를 찾고 있다
        이모지: 📚
        """

        try:
            response = self.llm_utils.get_llm_response(
                prompt, temperature=0.3, max_tokens=100
            )

            lines = response.strip().split('\n')
            description = action  # 기본값
            emoji = "🤔"  # 기본값

            for line in lines:
                if line.startswith("설명:"):
                    description = line.replace("설명:", "").strip()
                elif line.startswith("이모지:"):
                    emoji = line.replace("이모지:", "").strip()

        except Exception as e:
            print(f"[ActionExecutor] 행동 설명 생성 실패: {e}")
            description = f"{location}에서 {action}"
            emoji = "🤔"

        return description, emoji

    def _set_new_action(self, action, location, description, emoji, duration, start_time):
        """새로운 행동 설정"""
        self.current_action = action
        self.target_location = location
        self.action_description = description
        self.action_emoji = emoji
        self.action_duration = duration
        self.action_start_time = start_time

        print(f"[ActionExecutor] 새로운 행동: {emoji} {description} (@{location}, {duration}분)")

        # 메모리에 행동 기록
        self.npc.memory_manager.add_memory(
            'event',
            f"{self.npc.name}가 {location}에서 {action}를 시작했다",
            importance=5
        )

    def get_current_status(self):
        """현재 행동 상태 반환"""
        if not self.current_action:
            return {
                "action": "대기 중",
                "description": "할 일을 찾고 있음",
                "emoji": "🤔",
                "location": "알 수 없음",
                "progress": 0.0
            }

        # 진행률 계산
        if self.action_start_time and self.action_duration > 0:
            from time_manager import time_manager
            current_time = time_manager.get_current_time()
            elapsed_minutes = (current_time - self.action_start_time).total_seconds() / 60
            progress = min(1.0, elapsed_minutes / self.action_duration)
        else:
            progress = 0.0

        return {
            "action": self.current_action,
            "description": self.action_description,
            "emoji": self.action_emoji,
            "location": self.target_location,
            "progress": progress,
            "remaining_minutes": max(0, self.action_duration - elapsed_minutes) if self.action_start_time else 0
        }

    def handle_player_interaction(self, player_location, interaction_type="chat"):
        """플레이어 상호작용 처리"""
        print(f"[ActionExecutor] 플레이어 상호작용 처리: {interaction_type}")

        # 상호작용으로 인한 감정 변화
        self.npc.update_emotion("호기심")

        # 현재 행동 일시 중단
        if self.current_action:
            self.npc.memory_manager.add_memory(
                'event',
                f"플레이어와 {interaction_type} 상호작용으로 {self.current_action}를 중단했다",
                importance=7
            )

        # 대화 모드로 전환
        self._set_new_action(
            action="플레이어와 대화",
            location=player_location,
            description="플레이어와 대화 중",
            emoji="💬",
            duration=10,  # 기본 10분
            start_time=None  # 대화는 시간 제한 없음
        )

        return True

    def get_unity_movement_command(self):
        """Unity에 보낼 이동 명령 생성"""
        if not self.target_location:
            return None

        # 위치 문자열을 Unity 좌표로 변환 (예시)
        location_coordinates = self._location_to_coordinates(self.target_location)

        return {
            "npc_id": self.npc.name,
            "target_location": location_coordinates,
            "action_description": self.action_description,
            "emoji": self.action_emoji,
            "movement_speed": self._get_movement_speed()
        }

    def _location_to_coordinates(self, location):
        """위치 문자열을 Unity 좌표로 변환"""
        # 임시 좌표 매핑 (실제로는 Unity의 위치 시스템과 연동)
        coordinates_map = {
            "집:침실": {"x": 10, "z": 10},
            "집:부엌": {"x": 15, "z": 10},
            "집:거실": {"x": 12, "z": 8},
            "도서관:열람실": {"x": 50, "z": 30},
            "카페:휴게실": {"x": 30, "z": 20},
            "대학교:강의실": {"x": 70, "z": 40},
            "대학교:중앙광장": {"x": 60, "z": 35}
        }

        return coordinates_map.get(location, {"x": 0, "z": 0})

    def _get_movement_speed(self):
        """이동 속도 결정"""
        if "급하" in self.action_description or "서둘" in self.action_description:
            return "fast"
        elif "천천히" in self.action_description or "여유" in self.action_description:
            return "slow"
        else:
            return "normal"