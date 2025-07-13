# main_autonomous.py
"""
자율 NPC 시스템 메인 실행 스크립트
논문 기반의 generative agent 자율 행동 시스템
"""

import sys
import os
import argparse
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_autonomous import *
from llm_utils import LLM_Utils
from npc_agent_autonomous import AutonomousNpcAgent
from time_manager import time_manager
from server_autonomous import run_server


def check_dependencies():
    """필요한 의존성 확인"""
    print("🔍 시스템 의존성 확인 중...")

    try:
        import openai
        import fastapi
        import uvicorn
        import numpy
        print("✅ 모든 필수 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필수 패키지가 누락되었습니다: {e}")
        print("pip install -r requirements.txt 를 실행해주세요.")
        return False


def check_api_key():
    """OpenAI API 키 확인"""
    print("🔑 OpenAI API 키 확인 중...")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-api-key-here":
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("config_autonomous.py 파일에서 OPENAI_API_KEY를 설정해주세요.")
        return False

    # API 키 유효성 간단 테스트
    try:
        llm_utils = LLM_Utils(api_key=OPENAI_API_KEY)
        test_response = llm_utils.get_llm_response("안녕하세요", max_tokens=10)
        if "오류" in test_response or "ERROR" in test_response:
            print("❌ API 키가 유효하지 않을 수 있습니다.")
            return False
        print("✅ OpenAI API 키가 유효합니다.")
        return True
    except Exception as e:
        print(f"❌ API 키 테스트 실패: {e}")
        return False


def setup_memory_directories():
    """메모리 디렉토리 설정"""
    print("📁 메모리 디렉토리 설정 중...")

    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        print(f"✅ 메모리 디렉토리 생성: {MEMORY_DIR}")
        return True
    except Exception as e:
        print(f"❌ 메모리 디렉토리 생성 실패: {e}")
        return False


def create_test_npc(llm_utils):
    """테스트용 NPC 생성"""
    print("🤖 테스트 NPC 생성 중...")

    try:
        test_npc = AutonomousNpcAgent(
            name="이서아",
            persona="21살의 대학생. 시각 디자인을 전공하며 졸업 작품으로 고민이 많다. 성격은 내향적이지만 친근하고, 도움을 요청받으면 기꺼이 도와준다.",
            llm_utils=llm_utils
        )

        print(f"✅ NPC '{test_npc.name}' 생성 완료")
        return test_npc
    except Exception as e:
        print(f"❌ NPC 생성 실패: {e}")
        return None


def run_standalone_test():
    """독립 실행 테스트 모드"""
    print("🧪 독립 실행 테스트 모드 시작")
    print("=" * 50)

    # 시스템 체크
    if not check_dependencies() or not check_api_key() or not setup_memory_directories():
        return False

    # LLM 유틸리티 초기화
    llm_utils = LLM_Utils(api_key=OPENAI_API_KEY)

    # 테스트 NPC 생성
    npc = create_test_npc(llm_utils)
    if not npc:
        return False

    # 시간 관리자 시작
    print("⏰ 게임 시간 시스템 시작...")
    time_manager.set_time_speed(300)  # 5분에 1시간 (빠른 테스트)
    time_manager.start_time_flow()

    print(f"🕐 현재 게임 시간: {time_manager.get_time_str()}")

    # 초기 계획 생성 테스트
    print("📋 초기 일일 계획 생성 중...")
    current_time = time_manager.get_current_time()
    npc.planner.create_new_daily_plan(current_time)

    print("✅ 초기 계획 생성 완료:")
    print(npc.planner.get_schedule_summary())

    # 자율 행동 테스트
    print("\n🚶 자율 행동 테스트 중...")
    for i in range(5):
        print(f"\n--- 테스트 {i + 1}/5 ---")

        # 자율 업데이트
        npc.autonomous_update()

        # 현재 상태 출력
        status = npc.executor.get_current_status()
        print(f"행동: {status['emoji']} {status['action']}")
        print(f"설명: {status['description']}")
        print(f"위치: {status['location']}")
        print(f"감정: {npc.current_emotion}")

        # 시간 진행
        import time
        time.sleep(3)

    # 플레이어 상호작용 테스트
    print("\n💬 플레이어 상호작용 테스트...")
    test_messages = [
        "안녕하세요! 지금 뭐 하고 계세요?",
        "오늘 하루 어떻게 보내실 예정인가요?",
        "과제가 많이 힘든가요?"
    ]

    for msg in test_messages:
        print(f"\n플레이어: {msg}")
        response = npc.respond_to_player(msg, "도서관:열람실")
        print(f"{npc.name}: {response}")

    # 상호작용 종료
    npc.end_player_interaction()

    # 시간 시스템 정지
    time_manager.stop_time_flow()

    print("\n✅ 독립 실행 테스트 완료!")
    return True


def run_server_mode():
    """서버 모드 실행"""
    print("🌐 서버 모드 시작")
    print("=" * 50)

    # 시스템 체크
    if not check_dependencies() or not check_api_key() or not setup_memory_directories():
        return False

    print("🚀 FastAPI 서버를 시작합니다...")
    print("📝 서버 실행 후 다음 URL에서 테스트 가능:")
    print("   - 서버 상태: http://localhost:8000/")
    print("   - API 문서: http://localhost:8000/docs")
    print("   - NPC 목록: http://localhost:8000/npc/list")
    print("   - 시간 상태: http://localhost:8000/time/status")

    try:
        run_server(host="localhost", port=8000)
    except KeyboardInterrupt:
        print("\n🛑 서버가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")
        return False

    return True


def interactive_demo():
    """대화형 데모 모드"""
    print("🎮 대화형 데모 모드")
    print("=" * 50)

    # 시스템 체크
    if not check_dependencies() or not check_api_key() or not setup_memory_directories():
        return False

    # 초기화
    llm_utils = LLM_Utils(api_key=OPENAI_API_KEY)
    npc = create_test_npc(llm_utils)
    if not npc:
        return False

    # 시간 시스템 시작
    time_manager.set_time_speed(60)  # 1분에 1시간
    time_manager.start_time_flow()

    print(f"🕐 현재 게임 시간: {time_manager.get_time_str()}")

    # 초기 계획 생성
    current_time = time_manager.get_current_time()
    npc.planner.create_new_daily_plan(current_time)

    print(f"\n👋 안녕하세요! {npc.name}와 대화해보세요!")
    print("💡 명령어:")
    print("   - 'status': 현재 상태 확인")
    print("   - 'schedule': 오늘 일정 확인")
    print("   - 'time': 현재 시간 확인")
    print("   - 'exit': 종료")
    print("   - 그 외: 자유롭게 대화")

    try:
        while True:
            # 자율 업데이트
            npc.autonomous_update()

            # 사용자 입력
            user_input = input(f"\n[{time_manager.get_current_time().strftime('%H:%M')}] 입력: ").strip()

            if user_input.lower() == 'exit':
                print("👋 데모를 종료합니다.")
                break
            elif user_input.lower() == 'status':
                status = npc.get_status_for_unity()
                print(f"📊 {npc.name} 현재 상태:")
                print(f"   행동: {status['emoji']} {status['current_action']}")
                print(f"   설명: {status['description']}")
                print(f"   감정: {status['emotion']}")
                print(f"   위치: {status['location']}")
                print(f"   생각: {status['current_thought']}")
            elif user_input.lower() == 'schedule':
                print(f"📅 {npc.name}의 오늘 일정:")
                print(npc.planner.get_schedule_summary())
            elif user_input.lower() == 'time':
                print(f"🕐 현재 게임 시간: {time_manager.get_time_str()}")
            elif user_input:
                # 일반 대화
                response = npc.respond_to_player(user_input, "알 수 없음")
                print(f"{npc.name}: {response}")

                # 잠시 후 상호작용 종료
                import time
                time.sleep(2)
                npc.end_player_interaction()

    except KeyboardInterrupt:
        print("\n\n🛑 데모가 중단되었습니다.")

    finally:
        time_manager.stop_time_flow()

    return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="자율 NPC 시스템 - Generative Agent 기반 자율 행동 NPC"
    )
    parser.add_argument(
        "mode",
        choices=["test", "server", "demo"],
        help="실행 모드 선택: test(독립테스트), server(웹서버), demo(대화형데모)"
    )
    parser.add_argument(
        "--time-speed",
        type=float,
        default=60,
        help="게임 시간 배속 (기본: 60x, 1분에 1시간)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 활성화"
    )

    args = parser.parse_args()

    print("🎯 자율 NPC 시스템 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 실행 모드: {args.mode}")

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("🐛 디버그 모드 활성화")

    success = False

    try:
        if args.mode == "test":
            success = run_standalone_test()
        elif args.mode == "server":
            success = run_server_mode()
        elif args.mode == "demo":
            success = interactive_demo()

        if success:
            print("\n✅ 프로그램이 성공적으로 완료되었습니다.")
        else:
            print("\n❌ 프로그램 실행 중 오류가 발생했습니다.")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()