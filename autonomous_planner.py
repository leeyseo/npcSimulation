# autonomous_planner.py
import random


class AutonomousPlanner:
    """NPC의 자율적 일일 계획 수립을 담당하는 클래스 (논문의 planning 모듈 구현)"""

    def __init__(self, npc_agent, llm_utils):
        self.npc = npc_agent
        self.llm_utils = llm_utils

        # 계획 상태
        self.daily_schedule = []  # [['activity', duration_in_minutes], ...]
        self.daily_schedule_org = []  # 원본 시간별 스케줄
        self.current_action_index = 0
        self.last_planning_date = None

    def should_replan(self, current_time):
        """새로운 계획이 필요한지 확인"""
        current_date = current_time.date()

        # 첫 계획이거나 새로운 날인 경우
        if (not self.last_planning_date or
                self.last_planning_date != current_date or
                not self.daily_schedule):
            return True

        return False

    def generate_wake_up_hour(self):
        """기상 시간 생성 (논문 구현)"""
        prompt = f"""
        {self.npc.persona}에 대한 정보입니다.

        이 사람의 생활 패턴과 성격을 고려할 때, 평소 몇 시에 일어날까요?
        6시부터 10시 사이의 시간 중에서 가장 적절한 시간을 하나의 숫자로만 답해주세요.

        예시: 7
        """

        try:
            response = self.llm_utils.get_llm_response(
                prompt, temperature=0.1, max_tokens=10
            )
            wake_hour = int(response.strip())
            return max(6, min(10, wake_hour))  # 6-10시 사이로 제한
        except:
            return 7  # 기본값

    def generate_daily_plan_outline(self, wake_up_hour):
        """하루 전체 계획 개요 생성 (논문 구현)"""
        prompt = f"""
        {self.npc.persona}

        오늘은 {self.npc.conversation_manager.get_conversation_summary()}

        이 사람이 {wake_up_hour}시에 일어나서 하루 동안 할 주요 활동들을 시간순으로 나열해주세요.
        하루 일과를 4-6개의 주요 활동으로 나누어 시간과 함께 적어주세요.

        형식: 
        1. 6:00 AM - 기상 및 아침 루틴
        2. 8:00 AM - 아침식사
        3. ...

        일반적인 대학생의 일과를 기반으로 하되, 이 사람의 성격과 전공을 반영해주세요.
        """

        response = self.llm_utils.get_llm_response(
            prompt, temperature=0.3, max_tokens=200
        )

        # 응답에서 활동 목록 추출
        activities = []
        for line in response.split('\n'):
            if line.strip() and any(char.isdigit() for char in line):
                activities.append(line.strip())

        return activities if activities else [
            "7:00 AM - 기상 및 아침 루틴",
            "8:00 AM - 아침식사",
            "9:00 AM - 공부 및 과제",
            "12:00 PM - 점심식사",
            "2:00 PM - 휴식 및 개인시간",
            "6:00 PM - 저녁식사",
            "8:00 PM - 개인 취미 활동",
            "11:00 PM - 잠자리 준비"
        ]

    def generate_hourly_schedule(self, daily_plan, wake_up_hour):
        """시간별 세부 스케줄 생성 (논문 구현)"""
        prompt = f"""
        다음은 {self.npc.name}의 하루 일과 계획입니다:

        {chr(10).join(daily_plan)}

        이를 바탕으로 {wake_up_hour}시부터 24시간 동안의 시간별 활동을 생성해주세요.
        각 활동의 지속 시간(분)도 함께 적어주세요.

        형식: 활동명, 지속시간(분)

        예시:
        잠자기, 420
        기상 및 아침 루틴, 60
        아침식사, 30
        ...

        총 1440분(24시간)이 되도록 해주세요.
        """

        response = self.llm_utils.get_llm_response(
            prompt, temperature=0.2, max_tokens=300
        )

        # 응답 파싱
        schedule = []
        total_minutes = 0

        for line in response.split('\n'):
            if ',' in line and line.strip():
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        activity = parts[0].strip()
                        duration = int(parts[1].strip())
                        schedule.append([activity, duration])
                        total_minutes += duration
                except:
                    continue

        # 시간이 부족하면 수면 시간 추가
        if total_minutes < 1440:
            remaining = 1440 - total_minutes
            schedule.append(["잠자기", remaining])

        return schedule

    def create_new_daily_plan(self, current_time):
        """새로운 하루 계획 생성 (논문의 long-term planning)"""
        print(f"[AutonomousPlanner] {self.npc.name}의 새로운 하루 계획 생성 중...")

        # 1. 기상 시간 결정
        wake_up_hour = self.generate_wake_up_hour()
        print(f"[AutonomousPlanner] 기상 시간: {wake_up_hour}시")

        # 2. 하루 일과 개요 생성
        daily_plan_outline = self.generate_daily_plan_outline(wake_up_hour)
        print(f"[AutonomousPlanner] 일과 개요: {len(daily_plan_outline)}개 활동")

        # 3. 시간별 세부 스케줄 생성
        self.daily_schedule = self.generate_hourly_schedule(daily_plan_outline, wake_up_hour)
        self.daily_schedule_org = self.daily_schedule.copy()

        # 4. 현재 시간에 맞는 활동 인덱스 찾기
        self.current_action_index = self._find_current_action_index(current_time)

        # 5. 계획을 메모리에 저장
        self._save_plan_to_memory(current_time, daily_plan_outline)

        self.last_planning_date = current_time.date()

        print(f"[AutonomousPlanner] 계획 생성 완료: {len(self.daily_schedule)}개 활동")

    def _find_current_action_index(self, current_time):
        """현재 시간에 해당하는 활동 인덱스 찾기"""
        current_minutes = current_time.hour * 60 + current_time.minute
        elapsed_minutes = 0

        for i, (activity, duration) in enumerate(self.daily_schedule):
            if elapsed_minutes + duration > current_minutes:
                return i
            elapsed_minutes += duration

        return len(self.daily_schedule) - 1  # 마지막 활동

    def _save_plan_to_memory(self, current_time, daily_plan):
        """계획을 장기 기억에 저장"""
        plan_description = f"{self.npc.name}의 {current_time.strftime('%Y년 %m월 %d일')} 계획: "
        plan_description += ", ".join([plan.split(" - ")[-1] if " - " in plan else plan for plan in daily_plan[:3]])

        self.npc.memory_manager.add_memory(
            'thought',
            plan_description,
            importance=8
        )

    def get_current_activity(self, current_time):
        """현재 시간에 해당하는 활동 반환"""
        if not self.daily_schedule:
            return "대기 중", 30

        # 현재 활동 인덱스 업데이트
        self.current_action_index = self._find_current_action_index(current_time)

        if self.current_action_index < len(self.daily_schedule):
            return self.daily_schedule[self.current_action_index]

        return "대기 중", 30

    def get_schedule_summary(self):
        """스케줄 요약 반환"""
        if not self.daily_schedule:
            return "계획이 없습니다."

        summary = ""
        elapsed_minutes = 0

        for i, (activity, duration) in enumerate(self.daily_schedule):
            hour = elapsed_minutes // 60
            minute = elapsed_minutes % 60

            status = "→ " if i == self.current_action_index else "  "
            summary += f"{status}{hour:02d}:{minute:02d} - {activity} ({duration}분)\n"

            elapsed_minutes += duration

        return summary

    def modify_schedule_for_interaction(self, interaction_type, duration_minutes):
        """상호작용으로 인한 스케줄 수정"""
        print(f"[AutonomousPlanner] 스케줄 수정: {interaction_type} ({duration_minutes}분)")

        # 현재 활동 시간 단축 또는 연기
        if self.current_action_index < len(self.daily_schedule):
            current_activity, current_duration = self.daily_schedule[self.current_action_index]

            if current_duration > duration_minutes:
                # 현재 활동 시간 단축
                self.daily_schedule[self.current_action_index][1] -= duration_minutes
            else:
                # 현재 활동을 나중으로 연기
                self.daily_schedule.insert(
                    self.current_action_index + 1,
                    [current_activity, current_duration]
                )
                self.daily_schedule[self.current_action_index] = [interaction_type, duration_minutes]

    def should_interact_with_player(self, player_location, current_time):
        """플레이어와 상호작용할지 결정"""
        current_activity, _ = self.get_current_activity(current_time)

        # 바쁜 활동 중에는 상호작용 가능성 낮음
        busy_activities = ["시험", "중요한 과제", "수업", "잠자기"]
        if any(busy_word in current_activity for busy_word in busy_activities):
            return False, "바쁨"

        # 같은 장소에 있고 사회적 활동 중이면 상호작용 가능성 높음
        social_activities = ["휴식", "카페", "식사", "산책"]
        if any(social_word in current_activity for social_word in social_activities):
            return True, "사회적"

        # 기본적으로는 50% 확률
        return random.random() > 0.5, "보통"