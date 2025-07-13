# time_manager.py
import datetime
import threading
import time
from CoreSystemFiles.config import DEFAULT_TIME_SPEED, GAME_START_TIME


class TimeManager:
    """게임 시간 관리를 담당하는 클래스 (논문의 시간 시스템 구현)"""

    def __init__(self, start_time_str=GAME_START_TIME, time_speed=DEFAULT_TIME_SPEED):
        # 게임 시작 시간 설정 (예: "February 13, 2023, 07:00:00")
        self.start_time = datetime.datetime.strptime(start_time_str, "%B %d, %Y, %H:%M:%S")
        self.current_time = self.start_time
        self.time_speed = time_speed  # 실시간 대비 게임시간 배속 (1 = 실시간, 60 = 1분에 1시간)

        # 시간 흐름 제어
        self.is_running = False
        self.time_thread = None
        self.last_update = time.time()

        # 이벤트 콜백들
        self.time_callbacks = {
            'minute': [],
            'hour': [],
            'day': [],
            'new_day': []
        }

    def start_time_flow(self):
        """시간 흐름 시작"""
        if not self.is_running:
            self.is_running = True
            self.time_thread = threading.Thread(target=self._time_loop, daemon=True)
            self.time_thread.start()
            print(f"[TimeManager] 시간 흐름 시작: {self.get_time_str()}")

    def stop_time_flow(self):
        """시간 흐름 정지"""
        self.is_running = False
        if self.time_thread:
            self.time_thread.join()
        print("[TimeManager] 시간 흐름 정지")

    def _time_loop(self):
        """시간 흐름 메인 루프"""
        while self.is_running:
            current_real_time = time.time()
            elapsed_real_time = current_real_time - self.last_update

            # 게임 시간 계산 (실시간 * 배속)
            game_time_elapsed = elapsed_real_time * self.time_speed

            old_time = self.current_time
            self.current_time += datetime.timedelta(seconds=game_time_elapsed)

            # 시간 변화 이벤트 트리거
            self._trigger_time_events(old_time, self.current_time)

            self.last_update = current_real_time
            time.sleep(1)  # 1초마다 업데이트

    def _trigger_time_events(self, old_time, new_time):
        """시간 변화에 따른 이벤트 트리거"""
        # 분 변화
        if old_time.minute != new_time.minute:
            for callback in self.time_callbacks['minute']:
                callback(new_time)

        # 시 변화
        if old_time.hour != new_time.hour:
            for callback in self.time_callbacks['hour']:
                callback(new_time)

        # 일 변화
        if old_time.day != new_time.day:
            for callback in self.time_callbacks['day']:
                callback(new_time)

            # 새로운 날 이벤트 (첫날이 아닌 경우)
            if old_time.date() != datetime.date(2023, 2, 13):  # 첫날이 아닌 경우
                for callback in self.time_callbacks['new_day']:
                    callback(new_time)

    def register_callback(self, event_type, callback):
        """시간 이벤트 콜백 등록"""
        if event_type in self.time_callbacks:
            self.time_callbacks[event_type].append(callback)

    def get_current_time(self):
        """현재 게임 시간 반환"""
        return self.current_time

    def get_time_str(self):
        """현재 시간을 문자열로 반환"""
        return self.current_time.strftime("%B %d, %Y, %H:%M:%S")

    def get_curr_date_str(self):
        """현재 날짜를 문자열로 반환"""
        return self.current_time.strftime("%A %B %d")

    def is_new_day(self, last_time):
        """새로운 날인지 확인"""
        if not last_time:
            return "First day"

        if last_time.date() != self.current_time.date():
            return "New day"

        return False

    def minutes_since_start_of_day(self):
        """하루 시작부터 현재까지의 분 수"""
        return self.current_time.hour * 60 + self.current_time.minute

    def set_time_speed(self, speed):
        """시간 배속 설정"""
        self.time_speed = speed
        print(f"[TimeManager] 시간 배속 변경: {speed}x")


# 전역 시간 관리자 인스턴스
time_manager = TimeManager()