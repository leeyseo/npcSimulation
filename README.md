# NPC Agent 시뮬레이션

LLM 기반의 지능형 NPC 에이전트 시스템입니다. 메모리 관리, 지식 학습, 대화 맥락 이해를 통해 자연스러운 대화를 제공합니다.

## 프로젝트 구조

```
├── config.py              # 설정 및 상수
├── llm_utils.py           # LLM API 유틸리티
├── data_structures.py     # 메모리, 지식 데이터 구조
├── memory_manager.py      # 메모리 관리 시스템
├── conversation_manager.py # 대화 관리 시스템
├── npc_agent.py          # NPC 에이전트 메인 클래스
├── main.py               # 실행 파일
├── requirements.txt      # 의존성 패키지
└── README.md            # 프로젝트 설명서
```

## 주요 기능

### 1. 메모리 시스템
- **단기 메모리**: 최근 이벤트를 임시 저장
- **장기 메모리**: 중요한 정보를 영구 저장
- **메모리 요약**: 단기 메모리를 주기적으로 요약하여 장기 메모리로 이관
- **키워드 인덱싱**: 효율적인 메모리 검색을 위한 키워드 기반 인덱스

### 2. 지식 베이스
- **동적 학습**: 대화를 통해 새로운 지식 자동 습득
- **의미 기반 검색**: 임베딩을 활용한 유사도 기반 지식 검색
- **맥락 이해**: 대화 맥락에서 중요한 개념 자동 추출

### 3. 대화 관리
- **대화 버퍼**: 최근 대화 내용 저장
- **맥락 요약**: 대화 흐름의 핵심 요약
- **자연스러운 응답**: 메모리와 지식을 활용한 일관된 응답

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. API 키 설정
`config.py` 파일에서 OpenAI API 키를 설정하세요:
```python
OPENAI_API_KEY = "your-api-key-here"
```

### 3. 실행
```bash
python main.py
```

## 설정 옵션

`config.py`에서 다음 설정들을 조정할 수 있습니다:

- `SHORT_TERM_MAXLEN`: 단기 메모리 최대 길이
- `SUMMARY_WINDOW`: 메모리 요약 주기
- `SCORE_WEIGHTS`: 메모리 검색 점수 가중치
- `CONVERSATION_BUFFER_SIZE`: 대화 버퍼 크기

## 사용 예시

```
--- 1:1 NPC 상호작용 시뮬레이션 ---
이서아: 안녕하세요. 처음 뵙겠습니다.

Player > 안녕하세요! 저는 김민수라고 합니다.
이서아: 안녕하세요 민수님! 만나서 반갑습니다.

Player > 무엇을 공부하고 계신가요?
이서아: 저는 시각 디자인을 전공하고 있어요. 졸업 작품 때문에 요즘 고민이 많답니다.

Player > exit
대화를 종료합니다.
```

## 파일 구조 상세

### config.py
- 모든 설정값과 상수를 중앙 관리
- API 키, 모델 설정, 메모리 파라미터 등

### llm_utils.py
- OpenAI API 호출 래퍼
- 텍스트 생성과 임베딩 생성 기능

### data_structures.py
- Memory 클래스: 메모리 데이터 구조
- Knowledge 클래스: 지식 베이스 데이터 구조

### memory_manager.py
- 메모리 생성, 저장, 검색 관리
- 키워드 추출 및 인덱싱
- 단기/장기 메모리 관리

### conversation_manager.py
- 대화 버퍼 관리
- 대화 맥락 요약
- 대화 흐름 추적

### npc_agent.py
- NPC의 전체적인 행동 조율
- 각 컴포넌트 통합 관리
- 플레이어 입력 처리 및 응답 생성

## 확장 가능성

- **감정 시스템**: 대화에 따른 감정 변화 모델링
- **목표 시스템**: 동적 목표 설정 및 변경
- **리플렉션 시스템**: 메모리 기반 자기 성찰
- **다중 NPC**: 여러 NPC 간의 상호작용

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.