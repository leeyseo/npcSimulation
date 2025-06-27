import openai
import datetime
import random
import numpy as np
import uuid
import time
import re
from collections import defaultdict

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
            return f"(API 키 없음) 임시 응답."

        try:
            time.sleep(0.5)
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a thoughtful AI character. Please follow the user's instructions precisely."},
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

    def get_embedding(self, text: str) -> list[float]:
        """주어진 텍스트의 임베딩 벡터를 반환합니다."""
        if not self.api_key: return [0.0] * 1536
        try:
            if not text or not text.strip(): return [0.0] * 1536
            text = text.replace("\n", " ")
            response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {e}")
            return [0.0] * 1536


class Memory:
    """하나의 기억 단위를 나타냅니다. (From associative_memory.py)"""

    def __init__(self, memory_type: str, description: str, importance: int, embedding: list[float],
                 subject: str, predicate: str, object_str: str, keywords: set,
                 evidence_ids: list[str] = None):
        self.id = str(uuid.uuid4())
        self.type = memory_type
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.importance = importance
        self.embedding = embedding
        self.subject = subject
        self.predicate = predicate
        self.object = object_str
        self.keywords = keywords
        self.last_accessed = self.timestamp
        self.evidence_ids = evidence_ids if evidence_ids is not None else []

    def __repr__(self):
        return f"Memory(ID: {self.id[-6:]}, Type: {self.type}, SPO:({self.subject},{self.predicate},{self.object}), Desc: '{self.description}', Imp: {self.importance})"


class NpcAgent:
    """
    1대1 상호작용 NPC를 위한 통합 클래스.
    """

    def __init__(self, name: str, persona: str, llm_utils: LLM_Utils):
        self.name = name
        self.persona = persona
        self.llm_utils = llm_utils

        self.seq_event: list[Memory] = []
        self.seq_thought: list[Memory] = []

        self.kw_to_event: dict[str, list[Memory]] = defaultdict(list)
        self.kw_to_thought: dict[str, list[Memory]] = defaultdict(list)
        self.kw_strength: dict[str, int] = defaultdict(int)

        self.current_situation = "플레이어와 마주보고 대화하고 있다."
        self.current_emotion = "평온함"
        self.last_reflection_time = datetime.datetime.now()
        self.current_goal = "플레이어와 친해지고, 나의 고민에 대한 조언을 얻고 싶다."

        self.recency_decay = 0.995
        self.reflection_importance_sum = 0
        self.reflection_threshold = 100

        self.add_memory(memory_type='event', description=f"나의 이름은 '{name}'이다.", importance=10)
        self.add_memory(memory_type='event', description=f"나의 성격 및 설정: '{persona}'", importance=10)
        self.add_memory(memory_type='thought', description=f"[목표] 나의 현재 목표는 '{self.current_goal}'이다.", importance=9)

    def _analyze_query(self, query: str) -> tuple[tuple[str, str, str], set[str]]:
        """쿼리를 분석하여 SPO와 키워드를 추출합니다."""
        spo = self._extract_spo(query, is_query=True)
        keywords = self._extract_keywords(query)
        return spo, keywords

    def _extract_spo(self, description: str, is_query: bool = False) -> tuple[str, str, str]:
        """개선된 프롬프트로 SPO를 추출합니다."""
        subject_pronoun = "나는" if not is_query else "너는"
        if description.startswith(subject_pronoun):
            description = self.name + description[2:]

        prompt = f"""
        다음 문장을 (주어, 서술어, 목적어) 형태의 핵심 관계로 요약해줘.
        - 문장의 핵심 주체를 '주어'로, 행동이나 상태를 '서술어'로, 행동의 대상을 '목적어'로 요약해줘.
        - '플레이어' 또는 '{self.name}'을 주체로 명확히 해줘.
        - 예시: "안녕 나는 경우야" -> (플레이어, 자기소개하다, 경우)
        - 예시: "나는 일반수학 2를 듣고 있어" -> (플레이어, 수강하다, 일반수학 2)
        - 결과는 반드시 "(주어, 서술어, 목적어)" 형식의 튜플로만 출력해줘.

        문장: "{description}"
        결과:
        """
        response = self.llm_utils.get_llm_response(prompt, temperature=0.0, max_tokens=60)

        try:
            match = re.search(r'\((.*?)\)', response)
            if match:
                parts = tuple(p.strip().strip("'\"") for p in match.group(1).split(','))
                if len(parts) == 3: return parts
        except Exception:
            pass
        return (self.name if not is_query else "플레이어", "관련된", description[:15])

    def _extract_keywords(self, description: str) -> set[str]:
        """개선된 프롬프트로 키워드와 개념을 추출합니다."""
        prompt = f"""
        다음 문장에서 가장 중요한 핵심 키워드를 5개 이하로 추출해줘. 
        - 문장에 직접 나타난 단어뿐만 아니라, 문장이 함의하는 '개념'도 함께 추출해줘 (예: "나는 경우야" -> 이름, 자기소개).
        - 각 키워드는 쉼표로 구분해서 명사 형태로 출력해줘.

        문장: "{description}"
        키워드:
        """
        response = self.llm_utils.get_llm_response(prompt, temperature=0.1, max_tokens=50)
        return {kw.strip() for kw in response.split(',') if kw.strip()}

    def add_memory(self, memory_type: str, description: str, importance: int = -1, evidence_ids: list[str] = None):
        """구조화된 새로운 기억을 메모리 스트림에 추가합니다."""
        if importance == -1:
            importance_prompt = f"'{self.name}'의 입장에서 다음 사건의 중요도를 1(사소함)에서 10(매우 중요함) 사이의 정수 하나로 평가해줘. 다른 설명 없이 숫자만 출력해줘.\n\n사건: {description}\n\n중요도:"
            try:
                res = self.llm_utils.get_llm_response(importance_prompt, temperature=0.0, max_tokens=3)
                importance = int(res.strip())
            except (ValueError, TypeError):
                importance = 5

        embedding = self.llm_utils.get_embedding(description)
        subject, predicate, object_str = self._extract_spo(description)
        keywords = self._extract_keywords(description)

        new_memory = Memory(memory_type, description, importance, embedding, subject, predicate, object_str, keywords,
                            evidence_ids)

        if memory_type == 'event':
            self.seq_event.append(new_memory)
        elif memory_type == 'thought':
            self.seq_thought.append(new_memory)

        for kw in keywords:
            if memory_type == 'event':
                self.kw_to_event[kw].append(new_memory)
            elif memory_type == 'thought':
                self.kw_to_thought[kw].append(new_memory)
            self.kw_strength[kw] += importance

        self.reflection_importance_sum += importance
        print(f"DEBUG (Add Memory): {new_memory}, Keywords: {keywords}, (누적 중요도: {self.reflection_importance_sum})")

        if self._should_reflect(): self.perform_reflection()

    def retrieve_memories(self, query: str, top_k: int = 10) -> list[Memory]:
        """SPO와 키워드로 1차 필터링 후, 가중치 점수로 최종 기억을 검색합니다."""
        print(f"\nDEBUG (Retrieve): \"{query}\"와 관련된 기억 검색 중...")
        query_spo, query_keywords = self._analyze_query(query)
        query_embedding = self.llm_utils.get_embedding(query)

        all_memories = self.seq_event + self.seq_thought
        candidate_memories = set()

        for kw in query_keywords:
            candidate_memories.update(self.kw_to_event.get(kw, []))
            candidate_memories.update(self.kw_to_thought.get(kw, []))

        for mem in all_memories:
            if query_spo[0] != self.name and (
                    query_spo[0] == mem.subject or query_spo[0] == mem.object): candidate_memories.add(mem)
            if query_spo[2] != query[:15] and (
                    query_spo[2] == mem.subject or query_spo[2] == mem.object): candidate_memories.add(mem)

        if not candidate_memories: candidate_memories = set(all_memories)
        print(f"DEBUG (Retrieve): 1차 필터링 후 후보 기억 {len(candidate_memories)}개")

        scores = []
        for memory in candidate_memories:
            recency_score = pow(self.recency_decay,
                                (datetime.datetime.now() - memory.last_accessed).total_seconds() / 3600)
            importance_score = memory.importance / 10.0
            relevance_score = self._cosine_similarity(query_embedding, memory.embedding)

            final_score = recency_score + importance_score + (relevance_score * 1.5)
            scores.append((final_score, memory))

        scores.sort(key=lambda x: x[0], reverse=True)
        retrieved_memories = [memory for score, memory in scores[:top_k]]

        print(
            f"DEBUG (Retrieve): 최종 상위 기억 {len(retrieved_memories)}개:\n{[m.description for m in retrieved_memories]}\n")
        return retrieved_memories

    def _cosine_similarity(self, v1, v2):
        vec1, vec2 = np.array(v1), np.array(v2)
        norm_v1, norm_v2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / (norm_v1 * norm_v2) if norm_v1 != 0 and norm_v2 != 0 else 0.0

    def _should_reflect(self) -> bool:
        return self.reflection_importance_sum > self.reflection_threshold

    def perform_reflection(self):
        print(f"\n{'=' * 20} [성찰 시작] {'=' * 20}\n")
        self.reflection_importance_sum = 0

        all_memories = self.seq_event + self.seq_thought

        focal_points_prompt = f"다음은 '{self.name}'의 최근 기억들이다.\n---\n{[m.description for m in all_memories[-50:]]}\n---\n이 기억들에서 도출할 수 있는, 더 높은 수준의 통찰을 얻기 위한 핵심 질문 3가지를 만들어줘. 각 질문은 한 줄로 작성해."
        focal_points_response = self.llm_utils.get_llm_response(focal_points_prompt)
        focal_points = [p.strip() for p in focal_points_response.split('\n') if p.strip()]

        new_insights = []
        for point in focal_points:
            point = point.strip()
            if not point or (point.startswith(('1.', '2.', '3.')) and len(point) < 5): continue

            relevant_memories = self.retrieve_memories(point, top_k=7)
            evidence_ids = [m.id for m in relevant_memories]

            insight_prompt = f"너의 성격: {self.persona}\n성찰 주제(질문): {point}\n\n이 질문에 답하기 위해 참고할 관련 기억(증거)들:\n{[m.description for m in relevant_memories]}\n---\n이 증거들을 바탕으로, 위 질문에 대한 깊이 있는 답변(깨달음)을 한 문장으로 요약해줘."
            insight = self.llm_utils.get_llm_response(insight_prompt, temperature=0.8)
            new_insights.append(insight)

            self.add_memory(memory_type='thought', description=f"[성찰] {insight}", importance=8,
                            evidence_ids=evidence_ids)

        self._update_goal(new_insights)
        print(f"\n{'=' * 20} [성찰 종료] {'=' * 20}\n")

    def _update_goal(self, new_insights: list[str]):
        print(f"DEBUG (Plan): 목표를 재설정합니다...")
        latest_thoughts = [m.description for m in self.seq_thought][-5:]
        goal_prompt = f"너의 기본 성격: {self.persona}\n너의 현재 장기 목표: {self.current_goal}\n최근에 얻은 깨달음:\n{new_insights}\n\n이 깨달음들을 종합하여, 앞으로의 상호작용을 위한 새로운 장기 목표를 한 문장으로 설정해줘."
        new_goal = self.llm_utils.get_llm_response(goal_prompt, temperature=0.8)
        self.current_goal = new_goal
        self.add_memory(memory_type='thought', description=f"[목표 재설정] 새로운 목표는 '{new_goal}' 이다.", importance=9)

    def _update_emotion(self, last_interaction: str):
        print(f"DEBUG (Emotion): 감정을 되돌아봅니다...")
        emotion_prompt = f"너는 '{self.name}'({self.persona})이고 현재 감정은 '{self.current_emotion}'이야.\n방금 이런 대화를 했어:\n{last_interaction}\n\n이 대화 후 너의 감정은 어떻게 변했니? 다음 중 하나로만 대답해줘: [기쁨, 슬픔, 놀람, 호기심, 평온함, 혼란스러움, 공감, 실망]"
        new_emotion = self.llm_utils.get_llm_response(emotion_prompt, temperature=0.5, max_tokens=10).strip().replace(
            ".", "")
        if new_emotion in ["기쁨", "슬픔", "놀람", "호기심", "평온함", "혼란스러움", "공감", "실망"] and new_emotion != self.current_emotion:
            self.current_emotion = new_emotion
            print(f"DEBUG (Emotion): 감정이 '{self.current_emotion}'에서 '{new_emotion}'(으)로 변경됨.")

    def respond_to_player(self, player_input: str) -> str:
        relevant_memories = self.retrieve_memories(player_input)
        memory_context = "\n".join(
            [f"- {m.description} (SPO: {m.subject}, {m.predicate}, {m.object})" for m in relevant_memories])

        response_prompt = f"""
        너는 '{self.name}'({self.persona})이고 현재 감정은 '{self.current_emotion}'이야.

        ### 현재 상황 ###
        {self.current_situation}

        ### 너의 현재 목표 ###
        {self.current_goal}

        ### 너의 응답에 참고할만한 과거 기억들 (구조화된 정보 포함) ###
        {memory_context}

        ### 방금 일어난 일 ###
        플레이어가 방금 너에게 이렇게 말했어: "{player_input}"

        ### 지시문 ###
        위의 모든 정보(특히 너의 현재 상황, 감정, 목표)를 종합적으로 고려하여, 플레이어에게 할 가장 자연스러운 응답을 한 문장으로 생성해줘.
        """
        response = self.llm_utils.get_llm_response(response_prompt)

        self.add_memory(memory_type='event', description=f"플레이어가 나에게 '{player_input}'라고 말했다.")
        self.add_memory(memory_type='event', description=f"나는 플레이어에게 '{response}'라고 대답했다.")
        self._update_emotion(f"Player: {player_input}\n{self.name}: {response}")
        return response


if __name__ == '__main__':
    llm_util = LLM_Utils(api_key=OPENAI_API_KEY)
    npc_name = "이서아"
    npc_persona = ("21살의 대학생. 시각 디자인을 전공하고 있으며, 가끔은 내향적이지만 친한 친구들과 어울리는 것을 좋아한다. "
                   "최근에는 졸업 작품 준비로 고민이 많고, 새로운 영감을 찾고 싶어한다.")
    npc = NpcAgent(name=npc_name, persona=npc_persona, llm_utils=llm_util)

    print("--- 1:1 NPC 상호작용 시뮬레이션 v15 (지능적 태깅 및 검색) ---")
    print("NPC와 대화를 시작해보세요. (종료하려면 'exit' 또는 Ctrl+C 입력)")
    print("-" * 50)
    print(f"{npc.name}: 아, 안녕하세요. 처음 뵙겠습니다.")

    try:
        while True:
            player_input = input("Player > ")
            if player_input.lower() == "exit":
                break

            if not player_input.strip():
                print(f"{npc.name}: ...?")
                continue

            npc_response = npc.respond_to_player(player_input)
            print(f"{npc.name}: {npc_response}")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\n" + "-" * 50)
        print(f"{npc.name}: 안녕히 가세요. 오늘 대화 즐거웠어요.")
