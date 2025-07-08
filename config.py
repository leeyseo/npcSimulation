# config.py
# 설정 및 상수 정의

# OpenAI API 키
OPENAI_API_KEY = "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"

# 메모리 설정
SHORT_TERM_MAXLEN = 20          # 단기 큐 최대 길이
SUMMARY_WINDOW = 10             # 요약 주기
REFLECTION_THRESHOLD = 100      # 리플렉션 임계값

# 점수 가중치
SCORE_WEIGHTS = [0.5, 1.5, 2.0]  # [recency, relevance, importance]
RECENCY_DECAY = 0.995

# 파일 경로
MEMORY_DIR = "memory_room"
SHORT_TERM_FILE = "short_term.json"
LONG_TERM_FILE = "long_term.json"

# 대화 설정
CONVERSATION_BUFFER_SIZE = 10

# LLM 설정
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 150
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"