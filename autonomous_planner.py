# autonomous_planner.py
import random
import re


class AutonomousPlanner:
    """NPC의 자율적 일일 계획 수립을 담당하는 클래스 (장소 및 기억 기반으로 개선)"""

    def __init__(self, npc_agent, llm_utils):
        self.npc = npc_agent
        self.llm_utils = llm_utils

        # 계획 상태
        self.daily_schedule = []
        self.daily_schedule_org = []
        self.current_action_index = 0
        self.last_planning_date = None

        # Unity로부터 받을 이동 가능 장소 목록
        self.available_locations = []

    def set_available_locations(self, locations: list):
        """Unity 환경에서 사용 가능한 장소 목록을 설정합니다."""
        self.available_locations = locations
        print(f"[{self.npc.name}의 Planner] 사용 가능한 장소 목록 업데이트: {self.available_locations}")

    def should_replan(self, current_time):
        """새로운 계획이 필요한지 확인"""
        current_date = current_time.date()
        if (not self.last_planning_date or
                self.last_planning_date != current_date or
                not self.daily_schedule):
            return True
        return False

    def generate_wake_up_hour(self):
        """기상 시간 생성"""
        prompt = f"{self.npc.persona}의 성격과 생활 패턴을 고려할 때, 보통 몇 시에 일어날까요? 6~10 사이의 숫자 하나로만 답해주세요."
        try:
            response = self.llm_utils.get_llm_response(prompt, temperature=0.1, max_tokens=10)
            return max(6, min(10, int(response.strip())))
        except:
            return 7

    def generate_daily_plan_outline(self, wake_up_hour):
        """하루 전체 계획 개요 생성 (장소 및 기억 기반)"""

        # 1. 최근 기억 가져오기 (MemoryManager에 retrieve_recent_memories 함수가 있다고 가정)
        recent_memories = self.npc.memory_manager.retrieve_recent_memories(count=5)
        memory_summary = "\n".join([f"- {m.description}" for m in recent_memories])

        # 2. 사용 가능한 장소 목록을 문자열로 변환
        locations_str = ", ".join(self.available_locations) if self.available_locations else "지정된 장소가 없음"

        prompt = f"""
        당신은 {self.npc.name}의 하루 계획을 세우는 AI 조수입니다.

        ### NPC 정보 ###
        {self.npc.persona}

        ### 최근 기억 및 대화 요약 ###
        {memory_summary}
        {self.npc.conversation_manager.get_conversation_summary()}

        ### 이동 가능한 장소 목록 ###
        {locations_str}

        ### 지시사항 ###
        위의 NPC 정보, 최근 기억, 그리고 **이동 가능한 장소 목록만을 참고**하여 {wake_up_hour}시에 일어난 후의 하루 일과를 4-6개의 주요 활동으로 나누어 시간과 함께 나열해주세요.
        각 활동은 반드시 주어진 장소 목록 내에서 이루어져야 합니다.
        NPC의 성격, 전공, 그리고 최근 기억을 반영하여 계획을 세워주세요. 예를 들어, '졸업 작품'에 대한 고민이 기억에 있다면, 계획에 '졸업 작품 구상'과 같은 활동을 포함시키세요.

        형식:
        1. 8:00 AM - 활동 (장소: [장소 목록 중 하나])
        2. 10:00 AM - 활동 (장소: [장소 목록 중 하나])
        ...
        """

        response = self.llm_utils.get_llm_response(prompt, temperature=0.4, max_tokens=300)

        activities = [line.strip() for line in response.split('\n') if
                      line.strip() and any(char.isdigit() for char in line)]
        return activities if activities else ["8:00 AM - 하루 시작 (장소: 집:침실)"]

    def generate_hourly_schedule(self, daily_plan, wake_up_hour):
        """시간별 세부 스케줄 생성 (개선된 버전)"""
        locations_str = ", ".join(self.available_locations) if self.available_locations else "지정된 장소가 없음"

        # 일과 개요를 문자열로 변환
        plan_str = "\n".join(daily_plan)

        prompt = f"""
        다음은 {self.npc.name}의 하루 일과 개요입니다:
        {plan_str}

        이 개요를 바탕으로 하루 종일(24시간)의 세부 활동 계획을 만들어주세요.
        각 활동은 다음 장소 목록 중 한 곳에서 이루어져야 합니다: {locations_str}

        다음 형식으로 정확히 답해주세요 (각 줄은 "활동명, 지속시간"으로 구성):

        아침 루틴, 30
        아침 식사, 60
        졸업 작품 구상, 120
        점심 식사, 60
        휴식, 90
        공부, 180
        저녁 식사, 60
        개인 시간, 120
        잠자기, 480

        총 시간이 1440분(24시간)이 되도록 맞춰주세요.
        각 활동은 반드시 위에 주어진 장소에서 할 수 있는 것이어야 합니다.
        """

        try:
            response = self.llm_utils.get_llm_response(prompt, temperature=0.2, max_tokens=400)
            print(f"[AutonomousPlanner] LLM 응답: {response}")

            schedule = []
            total_minutes = 0

            # 응답을 줄 단위로 분석
            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 쉼표로 분리 시도
                if ',' in line:
                    try:
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 2:
                            activity = parts[0]
                            # 숫자만 추출
                            duration_str = re.findall(r'\d+', parts[1])
                            if duration_str:
                                duration = int(duration_str[0])
                                schedule.append([activity, duration])
                                total_minutes += duration
                                print(f"[AutonomousPlanner] 활동 추가: {activity} ({duration}분)")
                    except (ValueError, IndexError) as e:
                        print(f"[AutonomousPlanner] 파싱 오류: {line} - {e}")
                        continue

            # 시간이 부족하면 기본 활동들 추가
            if total_minutes < 1440:
                remaining = 1440 - total_minutes
                if remaining > 300:  # 5시간 이상 남으면 잠자기와 여유 시간 추가
                    schedule.append(["여유 시간", min(remaining - 480, 120)])
                    schedule.append(["잠자기", max(480, remaining - 120)])
                else:
                    schedule.append(["잠자기", remaining])

            # 시간이 초과되면 조정
            elif total_minutes > 1440:
                excess = total_minutes - 1440
                for i in range(len(schedule) - 1, -1, -1):
                    if excess <= 0:
                        break
                    if schedule[i][1] > 60:  # 60분 이상인 활동만 줄임
                        reduction = min(excess, schedule[i][1] - 30)
                        schedule[i][1] -= reduction
                        excess -= reduction

            print(f"[AutonomousPlanner] 최종 스케줄 개수: {len(schedule)}, 총 시간: {sum(s[1] for s in schedule)}분")

            # 최소한의 기본 스케줄 보장
            if not schedule:
                schedule = [
                    ["기상 및 아침 루틴", 60],
                    ["아침 식사", 60],
                    ["공부", 180],
                    ["점심 식사", 60],
                    ["휴식", 120],
                    ["개인 시간", 180],
                    ["저녁 식사", 60],
                    ["잠자기", 719]
                ]
                print("[AutonomousPlanner] 기본 스케줄 사용")

            return schedule

        except Exception as e:
            print(f"[AutonomousPlanner] 스케줄 생성 오류: {e}")
            # 오류 발생 시 기본 스케줄 반환
            return [
                ["기상 및 아침 루틴", 60],
                ["아침 식사", 60],
                ["공부", 180],
                ["점심 식사", 60],
                ["휴식", 120],
                ["개인 시간", 180],
                ["저녁 식사", 60],
                ["잠자기", 719]
            ]

    def create_new_daily_plan(self, current_time):
        """새로운 하루 계획 생성"""
        print(f"[AutonomousPlanner] {self.npc.name}의 새로운 하루 계획 생성 중...")

        if not self.available_locations:
            print(f"[AutonomousPlanner] 경고: 사용 가능한 장소 목록이 비어있습니다. 계획 생성을 건너뜁니다.")
            return

        wake_up_hour = self.generate_wake_up_hour()
        daily_plan_outline = self.generate_daily_plan_outline(wake_up_hour)
        self.daily_schedule = self.generate_hourly_schedule(daily_plan_outline, wake_up_hour)
        self.daily_schedule_org = self.daily_schedule.copy()
        self.current_action_index = self._find_current_action_index(current_time)
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