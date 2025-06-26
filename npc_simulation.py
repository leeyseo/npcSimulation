import openai
import datetime
import random
import os

# ####################################################################################
# TODO: 여기에 자신의 OpenAI API 키를 입력하세요.
# https://platform.openai.com/account/api-keys 에서 발급받을 수 있습니다.
# (예: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
# ####################################################################################
OPENAI_API_KEY = ""
# ####################################################################################

class LLM_Utils:
    """
    모든 LLM API 호출을 담당하는 유틸리티 클래스.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key or "sk-" not in self.api_key:
            print("---")
            print("### 경고: OpenAI API 키가 유효하지 않습니다. ###")
            print("코드를 계속 실행하지만, LLM 대신 임시 응답을 사용합니다.")
            print("정상적인 작동을 위해 OPENAI_API_KEY를 설정해주세요.")
            print("---")
            self.api_key = None
        else:
            openai.api_key = self.api_key

    def get_llm_response(self, prompt: str, temperature=0.7, max_tokens=150) -> str:
        """주어진 프롬프트를 OpenAI API로 보내고 응답을 받아옵니다."""
        if not self.api_key:
            return f"(API 키 없음) 임시 응답: 사용자의 입력에 대해 생각해볼게요."

        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini", # 혹은 "gpt-4o"
                messages=[
                    {"role": "system", "content": "You are a thoughtful AI character who responds in a single, natural sentence from your persona's perspective."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message.content
            return response.strip()
        except Exception as e:
            print(f"LLM API 호출 중 오류가 발생했습니다: {e}")
            return "죄송해요, 지금은 생각에 잠겨 대답하기 어렵네요."


class Memory:
    """하나의 기억 단위를 나타냅니다. (From associative_memory.py)"""
    def __init__(self, description: str, importance: int):
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.importance = importance
        self.last_accessed = self.timestamp

    def __repr__(self):
        return f"Memory('{self.description}', Importance: {self.importance})"

class NpcAgent:
    """
    1대1 상호작용 NPC를 위한 통합 클래스.
    기억, 성찰, 계획, 현재 상태를 모두 관리합니다.
    """
    def __init__(self, name: str, persona: str, llm_utils: LLM_Utils):
        # --- 기본 정보 (From scratch.py) ---
        self.name = name
        self.persona = persona
        self.llm_utils = llm_utils

        # --- 기억 모듈 (From associative_memory.py) ---
        self.memory_stream: list[Memory] = []
        
        # --- 현재 상태 (From scratch.py) ---
        self.current_emotion = "평온함"
        self.last_reflection_time = datetime.datetime.now()
        
        # --- NPC의 초기 자기인식 기억 ---
        self.add_memory(f"나의 이름은 '{name}'이다.", importance=10)
        self.add_memory(f"나의 성격 및 설정: '{persona}'", importance=10)

    def add_memory(self, description: str, importance: int = -1):
        """
        새로운 기억을 메모리 스트림에 추가합니다.
        중요도가 지정되지 않으면 LLM을 통해 평가합니다.
        (From perceive.py + associative_memory.py)
        """
        if importance == -1:
            importance_prompt = f"""
            '{self.name}'의 입장에서 다음 사건의 중요도를 1(사소함)에서 10(매우 중요함) 사이의 정수 하나로 평가해줘.
            다른 설명 없이 숫자만 출력해줘.
            
            사건: {description}
            
            중요도:
            """
            try:
                res = self.llm_utils.get_llm_response(importance_prompt, temperature=0.0, max_tokens=3)
                importance = int(res.strip())
            except ValueError:
                print(f"DEBUG (Warning): 중요도 평가 실패. 기본값 5를 사용합니다. (응답: {res})")
                importance = 5
        
        new_memory = Memory(description, importance)
        self.memory_stream.append(new_memory)
        print(f"DEBUG (Add Memory): {new_memory}")

        # 특정 조건 충족 시 성찰을 트리거
        if self._should_reflect():
            self.perform_reflection()

    def retrieve_memories(self, query: str, top_k: int = 5) -> list[Memory]:
        """
        주어진 쿼리와 관련된 기억을 검색합니다.
        현재는 최신성+중요도 기반이지만, 추후 벡터 임베딩 기반으로 고도화될 것입니다.
        (From retrieve.py)
        """
        # TODO: 임베딩 기반의 정교한 검색 로직 구현 (Recency, Importance, Relevance)
        return sorted(self.memory_stream, 
                      key=lambda m: (m.timestamp, m.importance), 
                      reverse=True)[:top_k]

    def _should_reflect(self) -> bool:
        """성찰을 시작해야 할지 결정합니다. (From reflect.py)"""
        # 최근 10개 기억의 중요도 합계가 특정 임계값을 넘고, 마지막 성찰 후 충분한 시간이 지났을 때
        if (datetime.datetime.now() - self.last_reflection_time).total_seconds() < 60:
             return False # 너무 잦은 성찰 방지 (최소 1분 간격)
             
        recent_importance_sum = sum(m.importance for m in self.memory_stream[-10:])
        if recent_importance_sum > 40: # 임계값 (예: 40)
            return True
        return False

    def perform_reflection(self):
        """기억을 바탕으로 새로운 통찰(성찰)을 생성합니다. (From reflect.py)"""
        print(f"\nDEBUG (Reflection): '{self.name}'이(가) 깊은 생각에 잠깁니다...\n")
        self.last_reflection_time = datetime.datetime.now()

        # 1. 성찰의 주제가 될 핵심 질문/주제 생성
        focal_points_prompt = f"""
        다음은 '{self.name}'의 최근 기억들이다.
        ---
        {[m.description for m in self.memory_stream[-15:]]}
        ---
        이 기억들에서 도출할 수 있는 가장 중요한 핵심 질문이나 주제 3가지를 요약해줘.
        """
        focal_points_str = self.llm_utils.get_llm_response(focal_points_prompt, temperature=0.5)

        # 2. 생성된 주제를 바탕으로 통찰 생성
        insight_prompt = f"""
        너의 성격: {self.persona}
        최근 너의 생각 주제: {focal_points_str}
        
        이 주제들을 바탕으로 너 자신이나 세상에 대해 새롭게 얻은 깨달음이나 결심이 있다면 한 문장으로 요약해줘.
        """
        insight = self.llm_utils.get_llm_response(insight_prompt)
        
        # 3. 생성된 통찰을 중요도 높게 하여 기억에 추가
        self.add_memory(f"[성찰] {insight}", importance=10)
        print() # 보기 편하게 한 줄 띄우기

    def respond_to_player(self, player_input: str) -> str:
        """플레이어의 입력에 반응합니다. (From plan.py, converse.py)"""
        self.add_memory(f"플레이어가 나에게 '{player_input}'라고 말했다.")
        
        relevant_memories = self.retrieve_memories(player_input)
        memory_context = "\n".join([f"- {m.description}" for m in relevant_memories])
        
        response_prompt = f"""
        너는 '{self.name}'(이)야. 너의 성격은 다음과 같아: {self.persona}
        현재 너의 감정은 '{self.current_emotion}'이야.

        너의 응답에 참고할만한 과거 기억들:
        {memory_context}
        
        상황: 플레이어가 방금 너에게 이렇게 말했어: "{player_input}"
        
        '{self.name}'으로서 플레이어에게 할 자연스러운 응답을 한 문장으로 대답해줘.
        """
        
        response = self.llm_utils.get_llm_response(response_prompt)
        
        self.add_memory(f"나는 플레이어에게 '{response}'라고 대답했다.")
        return response

# --- 메인 실행 코드 ---
if __name__ == '__main__':
    llm_util = LLM_Utils(api_key=OPENAI_API_KEY)
    
    npc = NpcAgent(name="카이", 
                   persona="늘 호기심이 많고 다정하며, 인간의 다양한 감정과 관계에 대해 배우고 싶어하는 AI.",
                   llm_utils=llm_util)
    
    print("--- 1:1 NPC 상호작용 시뮬레이션을 시작합니다. ---")
    print("NPC와 대화를 시작해보세요. (종료하려면 'exit' 또는 Ctrl+C 입력)")
    print("-" * 50)
    print(f"{npc.name}: 안녕하세요. 만나서 반가워요. 당신에 대해 알고 싶어요.")

    try:
        while True:
            player_input = input("Player: ")
            if player_input.lower() == "exit":
                break
            
            npc_response = npc.respond_to_player(player_input)
            print(f"{npc.name}: {npc_response}")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("-" * 50)
        print(f"{npc.name}: 안녕히 가세요! 당신과의 대화는 제 기억 속에 남을 거예요.")
