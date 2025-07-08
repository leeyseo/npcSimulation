# main.py
from config import OPENAI_API_KEY
from llm_utils import LLM_Utils
from npc_agent import NpcAgent


def main():
    """메인 실행 함수"""
    # API 키 확인
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk":
        print("!!! OpenAI API 키를 config.py 파일에 입력해주세요. !!!")
        return

    # LLM 유틸리티 초기화
    llm_utils = LLM_Utils(api_key=OPENAI_API_KEY)

    # NPC 에이전트 생성
    npc_agent = NpcAgent(
        name="이서아",
        persona="21살의 대학생. 시각 디자인을 전공하며 졸업 작품으로 고민이 많다.",
        llm_utils=llm_utils
    )

    # 초기 인사
    initial_greeting = "안녕하세요. 처음 뵙겠습니다."
    print(f"--- 1:1 NPC 상호작용 시뮬레이션 (LLM 기반 지식 검색) ---")
    print(f"{npc_agent.name}: {initial_greeting}")

    # 초기 인사를 대화에 추가
    npc_agent.conversation_manager.add_message(npc_agent.name, initial_greeting)
    npc_agent.conversation_manager.summarize_conversation()

    # 대화 루프
    while True:
        try:
            player_input = input("Player > ")

            # 종료 조건
            if player_input.lower() in ['exit', 'quit', '종료']:
                print("대화를 종료합니다.")
                break

            # 빈 입력 처리
            if not player_input.strip():
                print(f"{npc_agent.name}: ...?")
                continue

            # NPC 응답 생성
            response = npc_agent.respond_to_player(player_input)
            print(f"{npc_agent.name}: {response}")

        except KeyboardInterrupt:
            print("\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue


if __name__ == "__main__":
    main()