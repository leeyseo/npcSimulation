# memory_manager.py
import datetime
import json
import os
import numpy as np
from collections import defaultdict, deque
from data_structures import Memory, Knowledge
from config import (
    SHORT_TERM_MAXLEN, SUMMARY_WINDOW, MEMORY_DIR,
    SHORT_TERM_FILE, LONG_TERM_FILE, SCORE_WEIGHTS, RECENCY_DECAY
)


class MemoryManager:
    """메모리 관리를 담당하는 클래스"""

    def __init__(self, llm_utils, name: str):
        self.llm_utils = llm_utils
        self.name = name

        # 메모리 저장소
        self.seq_event = []
        self.seq_thought = []
        self.kw_to_event = defaultdict(list)
        self.kw_to_thought = defaultdict(list)
        self.kw_strength = defaultdict(int)

        # 지식 베이스
        self.knowledge_base: dict[str, Knowledge] = {}

        # 메모리 룸
        self.memory_dir = MEMORY_DIR
        os.makedirs(self.memory_dir, exist_ok=True)
        self.short_term_path = os.path.join(self.memory_dir, SHORT_TERM_FILE)
        self.long_term_path = os.path.join(self.memory_dir, LONG_TERM_FILE)

        self.short_term_memory_room = deque(maxlen=SHORT_TERM_MAXLEN)
        self.long_term_memory_room = []

        # 점수 가중치
        self.score_weights = np.array(SCORE_WEIGHTS)
        self.recency_decay = RECENCY_DECAY

    def set_persona_description(self, persona_desc: str):
        """NPC의 페르소나 설명을 설정"""
        self.persona_description = persona_desc

    def _extract_keywords(self, description: str) -> set[str]:
        """키워드 추출"""
        prompt = (
            "다음 문장에서 가장 중요한 핵심 키워드를 5개 이하로 추출해줘. "
            "문장이 함의하는 '개념'도 포함해줘(예: \"나는 경우야\" -> 이름, 자기소개). "
            f"쉼표로 구분해서 명사 형태로 출력해줘.\n\n문장: \"{description}\"\n키워드:"
        )
        response = self.llm_utils.get_llm_response(prompt, temperature=0.1, max_tokens=50)
        return {kw.strip() for kw in response.split(',') if kw.strip()}

    def add_memory(self, memory_type: str, description: str, importance: int = -1,
                   evidence_ids: list[str] = None):
        """새로운 메모리를 추가"""
        if importance == -1:
            # 중요도 계산 시 memory_type을 함께 전달
            importance = self._calculate_importance(description, memory_type)

        embedding = self.llm_utils.get_embedding(description)
        keywords = self._extract_keywords(description)
        new_memory = Memory(memory_type, description, importance, embedding, keywords, evidence_ids)

        # 메모리 저장
        if memory_type == 'event':
            self.seq_event.append(new_memory)
        else:
            self.seq_thought.append(new_memory)

        # 키워드 인덱싱
        for kw in keywords:
            if memory_type == 'event':
                self.kw_to_event[kw].append(new_memory)
            else:
                self.kw_to_thought[kw].append(new_memory)
            self.kw_strength[kw] += importance

        # 단기 메모리 룸 처리
        if memory_type == 'event':
            self.short_term_memory_room.append(new_memory)

            if len(self.short_term_memory_room) >= SUMMARY_WINDOW:
                self._summarize_short_term()
                self.short_term_memory_room.clear()

        self._save_memory_rooms()
        print(f"DEBUG (Add Memory): {new_memory}")

    def _calculate_importance(self, description: str, memory_type: str) -> int:
        """
        기억의 종류(memory_type)에 따라 다른 프롬프트를 사용하여 중요도를 계산합니다.
        (논문 프롬프트 참조: poignancy_event_v1.txt, poignancy_thought_v1.txt, poignancy_chat_v1.txt)
        """
        # 기본 프롬프트 구조
        prompt_template = """
          다음은 '{name}'에 대한 간략한 설명입니다.
          {persona_description}

          1점에서 10점까지의 척도에서, 다음 {memory_category}의 중요도를 평가해 주세요.
          1점은 '{mundane_example}'처럼 지극히 평범한 것이며, 10점은 '{poignant_example}'처럼 매우 중대한 것입니다.

          {memory_category_label}: {description}
          점수 (1에서 10 사이의 숫자 하나만 반환):
          """

        # 기억 종류별 설정값
        if memory_type == 'event':
            settings = {
                "memory_category": "사건",
                "memory_category_label": "사건",
                "mundane_example": "이를 닦거나 침대를 정리하는 것",
                "poignant_example": "이별이나 대학 합격"
            }
        elif memory_type == 'thought':
            settings = {
                "memory_category": "생각",
                "memory_category_label": "생각",
                "mundane_example": "설거지를 해야 한다",
                "poignant_example": "교수가 되고 싶다"
            }
        # 'chat'이나 'summary' 등 다른 타입도 'event'와 유사하게 처리
        else:
            settings = {
                "memory_category": "대화 내용",
                "memory_category_label": "대화",
                "mundane_example": "일상적인 아침 인사",
                "poignant_example": "이별에 대한 대화나 싸움"
            }

        # 최종 프롬프트 생성
        prompt = prompt_template.format(
            name=self.name,
            persona_description=self.persona_description,
            memory_category=settings["memory_category"],
            mundane_example=settings["mundane_example"],
            poignant_example=settings["poignant_example"],
            memory_category_label=settings["memory_category_label"],
            description=description
        )

        try:
            response = self.llm_utils.get_llm_response(prompt, temperature=0.0, max_tokens=3)
            importance = int(response)
            return max(1, min(10, importance))
        except (ValueError, TypeError):
            # LLM이 숫자가 아닌 다른 답변을 할 경우를 대비한 기본값
            return 5

    def _summarize_short_term(self):
        """단기 메모리를 요약하여 장기 메모리로 이관"""
        joined = "\n".join([m.description for m in self.short_term_memory_room])
        prompt = f"다음 사건들의 핵심을 한 문장으로 요약해줘:\n{joined}\n\n[요약]"
        summary_sentence = self.llm_utils.get_llm_response(prompt, temperature=0.3, max_tokens=60)

        embedding = self.llm_utils.get_embedding(summary_sentence)
        keywords = self._extract_keywords(summary_sentence)
        summary_mem = Memory("summary", summary_sentence, 8, embedding, keywords)
        self.long_term_memory_room.append(summary_mem)

    def _normalize_scores(self, scores: dict) -> dict:
        """점수 딕셔너리를 0과 1 사이로 정규화하는 도우미 함수"""
        if not scores:
            return {}

        min_val = min(scores.values())
        max_val = max(scores.values())
        range_val = max_val - min_val

        if range_val == 0:
            return {k: 0.5 for k in scores}  # 모든 값이 같으면 중간값인 0.5로 설정

        normalized_scores = {
            key: (val - min_val) / range_val
            for key, val in scores.items()
        }
        return normalized_scores

    def retrieve_memories(self, query: str, top_k: int = 5) -> list[Memory]:
        """
        관련 메모리 검색 (논문 로직 적용 버전)
        최근성(Recency), 중요도(Importance), 관련성(Relevance)을 종합하여 점수를 매깁니다.
        """
        print(f"\nDEBUG (Retrieve): \"{query}\"와 관련된 기억 검색 중...")

        # --- 1. 모든 기억 노드 수집 ---
        all_memories = (
                self.seq_event + self.seq_thought +
                list(self.short_term_memory_room) + self.long_term_memory_room
        )
        # 중복 제거 및 ID를 키로 하는 딕셔너리 생성
        mem_map = {mem.id: mem for mem in all_memories}
        if not mem_map:
            return []

        # --- 2. 세 가지 핵심 점수 계산 ---
        query_embedding = self.llm_utils.get_embedding(query)

        recency_scores = {}
        importance_scores = {}
        relevance_scores = {}

        now = datetime.datetime.now()
        for mem_id, mem in mem_map.items():
            # 최근성 점수 계산
            hours_since_access = (now - mem.last_accessed).total_seconds() / 3600
            recency_scores[mem_id] = pow(self.recency_decay, hours_since_access)

            # 중요도 점수 계산
            importance_scores[mem_id] = mem.importance

            # 관련성 점수 계산
            relevance_scores[mem_id] = self._cosine_similarity(query_embedding, mem.embedding)

        # --- 3. 점수 정규화 ---
        # 각 점수 셋을 0과 1 사이로 정규화하여 공평하게 비교할 수 있도록 만듭니다.
        norm_recency = self._normalize_scores(recency_scores)
        norm_importance = self._normalize_scores(importance_scores)
        norm_relevance = self._normalize_scores(relevance_scores)

        # --- 4. 최종 점수 계산 (가중 합산) ---
        # 가중치: [최근성, 관련성, 중요도]. 논문 예시를 참고하여 설정합니다.
        # 이 값들을 조절하여 NPC의 기억 성향을 바꿀 수 있습니다.
        weights = [0.5, 3.0, 2.0]

        final_scores = {}
        for mem_id in mem_map:
            final_scores[mem_id] = (
                    weights[0] * norm_recency.get(mem_id, 0) +
                    weights[1] * norm_relevance.get(mem_id, 0) +
                    weights[2] * norm_importance.get(mem_id, 0)
            )

        # --- 5. 최상위 기억 선택 및 반환 ---
        # 최종 점수에 따라 메모리 ID를 정렬
        sorted_mems = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

        # 상위 top_k개의 메모리 객체를 가져옵니다.
        retrieved_ids = [mem_id for mem_id, score in sorted_mems[:top_k]]
        retrieved = [mem_map[mem_id] for mem_id in retrieved_ids]

        # 인출된 기억의 마지막 접근 시간을 갱신
        for mem in retrieved:
            mem.last_accessed = datetime.datetime.now()

        print(f"DEBUG (Retrieve): 최종 상위 기억 {len(retrieved)}개:\n{[m.description for m in retrieved]}\n")
        return retrieved

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> list[str]:
        """지식 베이스에서 관련 지식 검색"""
        print(f"DEBUG (Knowledge Retrieve): '{query}' 관련 지식 검색 중...")
        if not self.knowledge_base:
            return []

        lookup_prompt = (
            "다음 문장에서 내가 나의 '지식 베이스'에서 그 의미를 찾아봐야 할 중요한 고유명사나 핵심 개념은 무엇이야? "
            "가장 중요한 단어들을 쉼표로 구분해서 알려줘.\n\n"
            f"문장: \"{query}\"\n\n찾아봐야 할 단어:"
        )
        response = self.llm_utils.get_llm_response(lookup_prompt, temperature=0.0, max_tokens=50)
        query_keywords = {kw.strip() for kw in response.split(',') if kw.strip()}

        print(f"DEBUG (Knowledge Retrieve): LLM이 선별한 검색 키워드 -> {query_keywords}")
        if not query_keywords:
            return []

        # 키워드와 지식 베이스 간 유사도 계산
        candidate = []
        for kw in query_keywords:
            kw_emb = self.llm_utils.get_embedding(kw)
            for know in self.knowledge_base.values():
                sim = self._cosine_similarity(kw_emb, know.embedding)
                candidate.append((sim, know))

        candidate.sort(key=lambda x: x[0], reverse=True)
        seen, results = set(), []
        for _, know in candidate:
            if know.concept not in seen:
                results.append(f"- {know.concept}: {know.description}")
                seen.add(know.concept)
                if len(results) >= top_k:
                    break

        print(f"DEBUG (Knowledge Retrieve): 검색된 지식 -> {results}")
        return results

    def learn_from_interaction(self, interaction: str) -> dict:
        """상호작용으로부터 새로운 지식 학습 (새로 학습한 지식 딕셔너리 반환)"""
        print(f"DEBUG (Knowledge): 새로운 지식을 학습합니다...")
        prompt = f"""
        다음은 '{self.name}'와(과) 플레이어 간의 대화와, '{self.name}'가 이미 알고 있는 지식 목록입니다.

        [대화 내용]
        {interaction}

        [이미 알고 있는 지식]
        {list(self.knowledge_base.keys())}

        [지시]
        위 대화에서 '{self.name}'가 '새롭게' 알게 된 중요한 사실을 JSON 객체로 추출해줘.
        - **고유명사:** 사람 이름, 장소, 특정 과목명 등.
        - **관계적 의미:** 일반적인 단어지만 이 대화의 맥락에서 특별한 의미를 갖게 된 경우.
        설명은 반드시 플레이어와의 관계를 중심으로 작성해야 합니다.

        - **좋은 예시 1 (고유명사):** 플레이어가 "저는 컴공을 전공하는 경우입니다" 라고 말했다면,
          결과는 {{"경우": "플레이어의 이름", "컴공": "플레이어가 전공하고 있는 학과"}} 이어야 합니다.
        - **좋은 예시 2 (관계적 의미):** 플레이어가 "제 졸업 작품은 저의 '흰고래'예요" 라고 말했다면,
          결과는 {{"흰고래": "플레이어가 자신의 어렵고 중요한 졸업 작품을 비유적으로 표현하는 말"}} 이어야 합니다.
        - **나쁜 예시 (일반 사실):** 플레이어가 "하늘은 파랗다" 라고 말했다면, 결과는 {{}} 이어야 합니다.

        새로 알게 된 사실이 없다면, 빈 JSON 객체 {{}}를 반환해.
        반드시 올바른 JSON 형식으로만 응답해줘. 다른 텍스트는 포함하지 말고.

        [JSON 출력]
        """
        response_str = self.llm_utils.get_llm_response(prompt, temperature=0.1, max_tokens=500, is_json=True)

        try:
            new_knowledge = json.loads(response_str)
            if new_knowledge:
                for concept, desc in new_knowledge.items():
                    if concept not in self.knowledge_base:
                        emb = self.llm_utils.get_embedding(concept)
                        self.knowledge_base[concept] = Knowledge(concept, desc, emb)
                        print(f"DEBUG (Knowledge): 새로운 지식 추가! -> {self.knowledge_base[concept]}")
                        print(f'thought', f"[지식 습득] '{concept}'은(는) '{desc}'라는 것을 알게 되었다.")

        except json.JSONDecodeError:
            print(f"DEBUG (Knowledge): 지식 추출 실패. 응답: {response_str}")

    def _cosine_similarity(self, v1, v2):
        """코사인 유사도 계산"""
        v1, v2 = np.array(v1), np.array(v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0

    def _save_memory_rooms(self):
        """메모리 룸을 파일로 저장"""

        def _to_dict(mem: Memory):
            return {
                "type": mem.type,
                "desc": mem.description,
                "imp": mem.importance,
                "ts": mem.timestamp.isoformat()
            }

        with open(self.short_term_path, "w", encoding="utf-8") as f:
            json.dump([_to_dict(m) for m in self.short_term_memory_room], f, ensure_ascii=False, indent=2)

        with open(self.long_term_path, "w", encoding="utf-8") as f:
            json.dump([_to_dict(m) for m in self.long_term_memory_room], f, ensure_ascii=False, indent=2)


    def retrieve_recent_memories(self, count: int = 5) -> list[Memory]:
        """
        가장 최근의 기억(이벤트 및 생각)을 지정된 수만큼 가져옵니다.
        """
        print(f"\nDEBUG (Retrieve Recent): 가장 최근 기억 {count}개 검색 중...")

        # 모든 이벤트와 생각 기억을 하나의 리스트로 합칩니다.
        all_memories = self.seq_event + self.seq_thought

        # 기억들을 생성 시간(timestamp)을 기준으로 최신순으로 정렬합니다.
        # memory.timestamp가 최신일수록 뒤에 오므로, reverse=True로 순서를 뒤집습니다.
        sorted_memories = sorted(all_memories, key=lambda mem: mem.timestamp, reverse=True)

        # 가장 최근의 기억을 count만큼 잘라서 반환합니다.
        recent_memories = sorted_memories[:count]

        print(f"DEBUG (Retrieve Recent): 검색된 최근 기억 {len(recent_memories)}개:\n{[m.description for m in recent_memories]}\n")
        return recent_memories

