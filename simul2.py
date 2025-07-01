import openai
import datetime
import random
import numpy as np
import uuid
import time
import re
import json
from collections import defaultdict, deque

# ####################################################################################
# OpenAI API 키
# ####################################################################################
OPENAI_API_KEY = ""


# ####################################################################################


# ==================================================================================
# LLM 유틸리티 클래스
# ==================================================================================
class LLM_Utils:
    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key or "sk-" not in self.api_key:
            self.api_key = None
            print("### 경고: OpenAI API 키가 유효하지 않습니다. ###")
        else:
            openai.api_key = self.api_key

    def get_llm_response(self, prompt: str, temperature=0.7, max_tokens=150, is_json=False) -> str:
        if not self.api_key: return f"(API 키 없음) 임시 응답."
        try:
            time.sleep(0.5)
            model = "gpt-4o-mini"
            messages = [{"role": "system",
                         "content": "You are a thoughtful AI character. Please follow the user's instructions precisely."}]
            if is_json: messages[0]["content"] += " You must respond in the requested JSON format."
            messages.append({"role": "user", "content": prompt})
            response_format = {"type": "json_object"} if is_json else None
            completion = openai.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens, response_format=response_format)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"죄송해요, 오류가 발생했어요: {e}"

    def get_embedding(self, text: str) -> list[float]:
        if not self.api_key: return [0.0] * 1536
        try:
            if not text or not text.strip(): return [0.0] * 1536
            text = text.replace("\n", " ")
            response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
            return response.data[0].embedding
        except Exception as e:
            return [0.0] * 1536


# ==================================================================================
# 기억 및 지식 구조 클래스
# ==================================================================================
class Memory:
    def __init__(self, memory_type: str, description: str, importance: int, embedding: list[float], keywords: set,
                 evidence_ids: list[str] = None):
        self.id, self.type, self.timestamp, self.description, self.importance, self.embedding, self.keywords, self.last_accessed, self.evidence_ids = str(
            uuid.uuid4()), memory_type, datetime.datetime.now(), description, importance, embedding, keywords, datetime.datetime.now(), evidence_ids if evidence_ids is not None else []

    def __repr__(self):
        return f"Memory(ID: {self.id[-6:]}, Type: {self.type}, Desc: '{self.description}', Imp: {self.importance})"


class Knowledge:
    def __init__(self, concept: str, description: str, embedding: list[float]):
        self.id, self.concept, self.description, self.embedding, self.timestamp = str(
            uuid.uuid4()), concept, description, embedding, datetime.datetime.now()

    def __repr__(self): return f"Knowledge(Concept: '{self.concept}', Desc: '{self.description}')"


# ==================================================================================
# NPC 에이전트 클래스
# ==================================================================================
class NpcAgent:
    def __init__(self, name: str, persona: str, llm_utils: LLM_Utils):
        self.name, self.persona, self.llm_utils = name, persona, llm_utils
        self.seq_event, self.seq_thought = [], []
        self.kw_to_event, self.kw_to_thought, self.kw_strength = defaultdict(list), defaultdict(list), defaultdict(int)
        self.knowledge_base: dict[str, Knowledge] = {}

        self.conversation_buffer = deque(maxlen=10)
        self.conversation_summary = "아직 대화를 시작하지 않았다."

        self.current_situation = "플레이어와 마주보고 대화하고 있다."
        self.current_emotion, self.current_goal = "평온함", "플레이어와 친해지고, 나의 고민에 대한 조언을 얻고 싶다."

        self.recency_w, self.relevance_w, self.importance_w = 1.0, 1.0, 1.0
        self.score_weights = np.array([0.5, 1.5, 2.0])
        self.recency_decay = 0.995
        self.reflection_importance_sum, self.reflection_threshold = 0, 100

        self.add_memory('event', f"나의 이름은 '{name}'이다.", 10)
        self.add_memory('event', f"나의 성격 및 설정: '{persona}'", 10)
        self.add_memory('thought', f"[목표] 나의 현재 목표는 '{self.current_goal}'이다.", 9)

    def _extract_keywords(self, description: str) -> set[str]:
        prompt = f"다음 문장에서 가장 중요한 핵심 키워드를 5개 이하로 추출해줘. 문장이 함의하는 '개념'도 포함해줘(예: \"나는 경우야\" -> 이름, 자기소개). 쉼표로 구분해서 명사 형태로 출력해줘.\n\n문장: \"{description}\"\n키워드:"
        response = self.llm_utils.get_llm_response(prompt, temperature=0.1, max_tokens=50)
        return {kw.strip() for kw in response.split(',') if kw.strip()}

    def _learn_from_interaction(self, last_interaction: str):
        print(f"DEBUG (Knowledge): 새로운 지식을 학습합니다...")
        prompt = f"""
        다음은 '{self.name}'와(과) 플레이어 간의 대화와, '{self.name}'가 이미 알고 있는 지식 목록입니다.

        [대화 내용]
        {last_interaction}

        [이미 알고 있는 지식]
        {list(self.knowledge_base.keys())}

        [지시]
        위 대화에서 '{self.name}'가 '새롭게' 알게 된 중요한 사실을 JSON 객체로 추출해줘.
        - **고유명사:** 사람 이름, 장소, 특정 과목명 등.
        - **관계적 의미:** 일반적인 단어지만 이 대화의 맥락에서 특별한 의미를 갖게 된 경우.
        설명은 반드시 플레이어와의 관계를 중심으로 작성해야 합니다.

        - **좋은 예시 1 (고유명사):** 플레이어가 "저는 컴공을 전공하는 경우입니다" 라고 말했다면, 결과는 {{"경우": "플레이어의 이름", "컴공": "플레이어가 전공하고 있는 학과"}} 이어야 합니다.
        - **좋은 예시 2 (관계적 의미):** 플레이어가 "제 졸업 작품은 저의 '흰고래'예요" 라고 말했다면, 결과는 {{"흰고래": "플레이어가 자신의 어렵고 중요한 졸업 작품을 비유적으로 표현하는 말"}} 이어야 합니다.
        - **나쁜 예시 (일반 사실):** 플레이어가 "하늘은 파랗다" 라고 말했다면, 결과는 {{}} 이어야 합니다.

        새로 알게 된 사실이 없다면, 빈 JSON 객체 {{}}를 반환해.

        [JSON 출력]
        """
        response_str = self.llm_utils.get_llm_response(prompt, temperature=0.1, max_tokens=500, is_json=True)
        try:
            new_knowledge_dict = json.loads(response_str)
            if new_knowledge_dict:
                for concept, description in new_knowledge_dict.items():
                    if concept not in self.knowledge_base:
                        embedding = self.llm_utils.get_embedding(concept)
                        self.knowledge_base[concept] = Knowledge(concept, description, embedding)
                        print(f"DEBUG (Knowledge): 새로운 지식 추가! -> {self.knowledge_base[concept]}")
                        print('thought', f"[지식 습득] '{concept}'은(는) '{description}'라는 것을 알게 되었다.", 7)
        except json.JSONDecodeError:
            print(f"DEBUG (Knowledge): 지식 추출 실패. 응답: {response_str}")

    def add_memory(self, memory_type: str, description: str, importance: int = -1, evidence_ids: list[str] = None):
        if importance == -1:
            try:
                importance = int(self.llm_utils.get_llm_response(
                    f"'{self.name}'의 입장에서 다음 사건의 중요도를 1~10 사이 정수로 평가해줘: '{description}'", 0.0, 3))
            except:
                importance = 5
        embedding = self.llm_utils.get_embedding(description)
        keywords = self._extract_keywords(description)
        new_memory = Memory(memory_type, description, importance, embedding, keywords, evidence_ids)

        if memory_type == 'event':
            self.seq_event.append(new_memory)
        else:
            self.seq_thought.append(new_memory)

        for kw in keywords:
            if memory_type == 'event':
                self.kw_to_event[kw].append(new_memory)
            else:
                self.kw_to_thought[kw].append(new_memory)
            self.kw_strength[kw] += importance

        self.reflection_importance_sum += importance
        print(f"DEBUG (Add Memory): {new_memory}")
        if self._should_reflect(): self.perform_reflection()

    def retrieve_memories(self, query: str, top_k: int = 5) -> list[Memory]:
        """논문 수준의 가중치 기반 기억 검색 시스템."""
        print(f"\nDEBUG (Retrieve): \"{query}\"와 관련된 기억 검색 중...")
        query_keywords = self._extract_keywords(query)
        print(f"DEBUG (Retrieve): 검색 키워드 -> {query_keywords}")
        query_embedding = self.llm_utils.get_embedding(query)

        all_memories = self.seq_event + self.seq_thought
        candidate_memories = set()

        for kw in query_keywords:
            candidate_memories.update(self.kw_to_event.get(kw, []))
            candidate_memories.update(self.kw_to_thought.get(kw, []))

        if not candidate_memories: candidate_memories = set(all_memories)

        scores = []
        for memory in candidate_memories:
            recency_score = pow(self.recency_decay,
                                (datetime.datetime.now() - memory.last_accessed).total_seconds() / 3600)
            importance_score = memory.importance / 10.0
            relevance_score = self._cosine_similarity(query_embedding, memory.embedding)

            score_vector = np.array([recency_score, relevance_score, importance_score])
            final_score = np.dot(score_vector, self.score_weights)
            scores.append((final_score, memory))

        scores.sort(key=lambda x: x[0], reverse=True)
        retrieved_memories = [memory for score, memory in scores[:top_k]]

        print(
            f"DEBUG (Retrieve): 최종 상위 기억 {len(retrieved_memories)}개:\n{[m.description for m in retrieved_memories]}\n")
        return retrieved_memories

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> list[str]:
        """지식 베이스에서 LLM이 선별한 키워드로 관련 지식을 검색합니다."""
        print(f"DEBUG (Knowledge Retrieve): '{query}' 관련 지식 검색 중...")
        if not self.knowledge_base: return []

        lookup_prompt = f"다음 문장에서 내가 나의 '지식 베이스'에서 그 의미를 찾아봐야 할 중요한 고유명사나 핵심 개념은 무엇이야? 가장 중요한 단어들을 쉼표로 구분해서 알려줘.\n\n문장: \"{query}\"\n\n찾아봐야 할 단어:"
        query_keywords = {kw.strip() for kw in
                          self.llm_utils.get_llm_response(lookup_prompt, temperature=0.0, max_tokens=50).split(',') if
                          kw.strip()}
        print(f"DEBUG (Knowledge Retrieve): LLM이 선별한 검색 키워드 -> {query_keywords}")
        if not query_keywords: return []

        candidate_knowledge = []
        for kw in query_keywords:
            kw_embedding = self.llm_utils.get_embedding(kw)
            for knowledge_item in self.knowledge_base.values():
                similarity = self._cosine_similarity(kw_embedding, knowledge_item.embedding)
                candidate_knowledge.append((similarity, knowledge_item))

        candidate_knowledge.sort(key=lambda x: x[0], reverse=True)

        seen_concepts, retrieved_knowledge = set(), []
        for _, knowledge in candidate_knowledge:
            if knowledge.concept not in seen_concepts:
                retrieved_knowledge.append(f"- {knowledge.concept}: {knowledge.description}")
                seen_concepts.add(knowledge.concept)
                if len(retrieved_knowledge) >= top_k: break

        print(f"DEBUG (Knowledge Retrieve): 검색된 지식 -> {retrieved_knowledge}")
        return retrieved_knowledge

    def _summarize_conversation(self):
        print("DEBUG (Context): 현재 대화 맥락을 요약합니다...")
        buffer_str = "\n".join([f"{speaker}: {text}" for speaker, text in self.conversation_buffer])
        prompt = f"다음은 최근 대화 기록이야. 이 대화의 현재 주제와 흐름을 한 문장으로 요약해줘.\n\n[대화 기록]\n{buffer_str}\n\n[요약]"
        summary = self.llm_utils.get_llm_response(prompt, temperature=0.3, max_tokens=100)
        self.conversation_summary = summary
        print(f"DEBUG (Context): 요약된 현재 대화 맥락 -> {self.conversation_summary}")

    def _cosine_similarity(self, v1, v2):
        vec1, vec2 = np.array(v1), np.array(v2)
        norm_v1, norm_v2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / (norm_v1 * norm_v2) if norm_v1 != 0 and norm_v2 != 0 else 0.0

    def respond_to_player(self, player_input: str) -> str:
        self.conversation_buffer.append(("Player", player_input))

        relevant_memories = self.retrieve_memories(player_input)
        relevant_knowledge = self.retrieve_knowledge(player_input)

        memory_context = "\n".join([f"- {m.description}" for m in relevant_memories])
        knowledge_context = "\n".join(relevant_knowledge)

        response_prompt = f"""
        너는 '{self.name}'({self.persona})이고 현재 감정은 '{self.current_emotion}'이야.

        ### 현재 대화의 핵심 흐름 ###
        {self.conversation_summary}

        ### 너가 알고 있는 사실 (지식 베이스) ###
        {knowledge_context}

        ### 너의 장기 기억 (과거 사건 및 생각) ###
        {memory_context}

        ### 방금 일어난 일 ###
        플레이어가 방금 너에게 이렇게 말했어: "{player_input}"

        ### 지시문 ###
        위의 모든 정보(특히 '현재 대화의 핵심 흐름')를 가장 중요하게 고려하여, 플레이어에게 할 가장 자연스러운 다음 응답을 한 문장으로 생성해줘.
        """
        response = self.llm_utils.get_llm_response(response_prompt)

        self.add_memory('event', f"플레이어가 나에게 '{player_input}'라고 말했다.")
        self.add_memory('event', f"나는 플레이어에게 '{response}'라고 대답했다.")

        self.conversation_buffer.append((self.name, response))
        self._summarize_conversation()
        self._learn_from_interaction(f"Player: {player_input}\n{self.name}: {response}")
        return response

    def _should_reflect(self):
        return False

    def perform_reflection(self):
        pass

    def _update_goal(self, insights):
        pass


if __name__ == '__main__':
    if not OPENAI_API_KEY:
        print("!!! OpenAI API 키를 파일 상단에 입력해주세요. !!!")
    else:
        llm = LLM_Utils(api_key=OPENAI_API_KEY)
        npc_agent = NpcAgent(name="이서아",
                             persona="21살의 대학생. 시각 디자인을 전공하며 졸업 작품으로 고민이 많다.",
                             llm_utils=llm)

        initial_greeting = "안녕하세요. 처음 뵙겠습니다."
        print(f"--- 1:1 NPC 상호작용 시뮬레이션 (LLM 기반 지식 검색) ---")
        print(f"{npc_agent.name}: {initial_greeting}")
        npc_agent.conversation_buffer.append((npc_agent.name, initial_greeting))
        npc_agent._summarize_conversation()

        while True:
            p_input = input("Player > ")
            if p_input.lower() == 'exit': break
            if not p_input.strip(): print(f"{npc_agent.name}: ...?"); continue
            response = npc_agent.respond_to_player(p_input)
            print(f"{npc_agent.name}: {response}")
