# config_autonomous.py
# 기존 설정들 + 자율 행동 시스템 설정

# OpenAI API 키
OPENAI_API_KEY = "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"

# 메모리 설정
SHORT_TERM_MAXLEN = 50          # 단기 큐 최대 길이
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

# === 새로운 자율 행동 시스템 설정 ===

# 시간 관리 설정
GAME_START_TIME = "February 13, 2023, 07:00:00"  # 게임 시작 시간
DEFAULT_TIME_SPEED = 60  # 시간 배속 (60 = 1분에 1시간)
TIME_UPDATE_INTERVAL = 1  # 시간 업데이트 간격 (초)

# 자율 행동 설정
AUTONOMOUS_UPDATE_INTERVAL = 60  # 자율 행동 업데이트 간격 (초)
DEFAULT_ACTION_DURATION = 30  # 기본 행동 지속 시간 (분)
MIN_ACTION_DURATION = 5  # 최소 행동 지속 시간 (분)
MAX_ACTION_DURATION = 180  # 최대 행동 지속 시간 (분)

# 계획 수립 설정
DAILY_PLANNING_HOUR = 6  # 일일 계획 수립 시간
MIN_WAKE_HOUR = 6  # 최소 기상 시간
MAX_WAKE_HOUR = 10  # 최대 기상 시간
SLEEP_HOUR = 23  # 기본 취침 시간

# 상호작용 설정
INTERACTION_PROBABILITY = 0.7  # 플레이어와 상호작용할 기본 확률
BUSY_INTERACTION_PROBABILITY = 0.3  # 바쁠 때 상호작용 확률
SOCIAL_INTERACTION_PROBABILITY = 0.9  # 사회적 활동 중 상호작용 확률

# 감정 시스템 설정
DEFAULT_EMOTION = "평온함"
EMOTION_DECAY_TIME = 120  # 감정 지속 시간 (분)
EMOTION_UPDATE_THRESHOLD = 5  # 감정 변화 임계값

# 위치 시스템 설정
DEFAULT_LOCATION = "대학교:중앙광장"
MOVEMENT_SPEED = {
    "slow": 1.0,
    "normal": 2.0,
    "fast": 3.5
}

# Unity 통신 설정
UNITY_UPDATE_RATE = 30  # Unity 업데이트 주기 (초)
MAX_UNITY_QUEUE_SIZE = 100  # Unity 명령 큐 최대 크기

# 로깅 설정
ENABLE_DEBUG_LOGS = True
LOG_AUTONOMOUS_ACTIONS = True
LOG_MEMORY_OPERATIONS = True
LOG_TIME_EVENTS = True

# 성능 최적화 설정
MAX_MEMORY_SEARCH_RESULTS = 10  # 메모리 검색 최대 결과 수
MEMORY_CLEANUP_INTERVAL = 3600  # 메모리 정리 간격 (초)
MAX_DAILY_MEMORIES = 200  # 하루 최대 기억 수

# 행동 카테고리 정의
ACTION_CATEGORIES = {
    "필수": ["잠자기", "식사", "화장실"],
    "학업": ["공부", "과제", "수업", "도서관"],
    "사회": ["대화", "휴식", "카페", "만남"],
    "개인": ["취미", "운동", "쇼핑", "산책"],
    "기타": ["이동", "대기", "생각"]
}

# 위치별 가능한 활동 정의
LOCATION_ACTIVITIES = {
    "집:침실": ["잠자기", "휴식", "옷 갈아입기"],
    "집:부엌": ["요리", "식사", "설거지"],
    "집:거실": ["TV 시청", "휴식", "독서"],
    "도서관:열람실": ["공부", "과제", "독서", "자료 검색"],
    "카페:휴게실": ["휴식", "간식", "대화", "폰 보기"],
    "대학교:강의실": ["수업", "발표", "시험"],
    "대학교:중앙광장": ["이동", "만남", "산책"],
    "체육관:운동실": ["운동", "헬스", "샤워"],
    "상점:매장": ["쇼핑", "구매", "둘러보기"],
    "공원:산책로": ["산책", "조깅", "휴식"]
}

# 시간대별 활동 확률
HOURLY_ACTIVITY_WEIGHTS = {
    6: {"잠자기": 0.8, "기상": 0.2},
    7: {"아침루틴": 0.6, "아침식사": 0.4},
    8: {"아침식사": 0.3, "이동": 0.4, "수업": 0.3},
    9: {"수업": 0.7, "공부": 0.3},
    12: {"점심식사": 0.8, "휴식": 0.2},
    18: {"저녁식사": 0.6, "휴식": 0.4},
    22: {"개인시간": 0.7, "잠자리준비": 0.3},
    23: {"잠자기": 0.9, "개인시간": 0.1}
}

# NPC 성격별 행동 선호도
PERSONALITY_PREFERENCES = {
    "내향적": {
        "선호": ["독서", "혼자 공부", "집에서 휴식"],
        "회피": ["큰 모임", "시끄러운 장소"]
    },
    "외향적": {
        "선호": ["대화", "모임", "카페", "운동"],
        "회피": ["혼자 있기", "조용한 활동"]
    },
    "성실한": {
        "선호": ["공부", "과제", "계획적 활동"],
        "회피": ["늦은 기상", "게으름"]
    },
    "창의적": {
        "선호": ["그림", "음악", "자유시간"],
        "회피": ["단조로운 활동", "엄격한 일정"]
    }
}