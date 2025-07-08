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
            importance = self._calculate_importance(description)

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

    def _calculate_importance(self, description: str) -> int:
        """중요도 계산"""
        try:
            importance = int(self.llm_utils.get_llm_response(
                f"'{self.name}'의 입장에서 다음 사건의 중요도를 1~10 사이 정수로 평가해줘: '{description}'",
                0.0, 3))
            return max(1, min(10, importance))
        except Exception:
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

    def retrieve_memories(self, query: str, top_k: int = 5) -> list[Memory]:
        """관련 메모리 검색"""
        print(f"\nDEBUG (Retrieve): \"{query}\"와 관련된 기억 검색 중...")
        query_keywords = self._extract_keywords(query)
        print(f"DEBUG (Retrieve): 검색 키워드 -> {query_keywords}")
        query_embedding = self.llm_utils.get_embedding(query)

        # 모든 메모리 수집
        all_memories = (
                list(self.short_term_memory_room) + self.long_term_memory_room +
                self.seq_event + self.seq_thought
        )

        # 키워드 기반 후보 선별
        candidate_memories = set()
        for kw in query_keywords:
            candidate_memories.update(self.kw_to_event.get(kw, []))
            candidate_memories.update(self.kw_to_thought.get(kw, []))

        if not candidate_memories:
            candidate_memories = set(all_memories)

        # 점수 계산 및 정렬
        scores = []
        for mem in candidate_memories:
            recency_score = pow(self.recency_decay,
                                (datetime.datetime.now() - mem.last_accessed).total_seconds() / 3600)
            importance_score = mem.importance / 10.0
            relevance_score = self._cosine_similarity(query_embedding, mem.embedding)
            final_score = np.dot(np.array([recency_score, relevance_score, importance_score]), self.score_weights)
            scores.append((final_score, mem))

        scores.sort(key=lambda x: x[0], reverse=True)
        retrieved = [m for _, m in scores[:top_k]]
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
                        self.add_memory('thought', f"[지식 습득] '{concept}'은(는) '{desc}'라는 것을 알게 되었다.", 7)
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