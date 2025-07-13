from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from threading import Lock

from config import OPENAI_API_KEY
from llm_utils import LLM_Utils
from npc_agent import NpcAgent


# 요청/응답 모델
class ChatRequest(BaseModel):
    npc_id: str
    player_message: str
    player_name: Optional[str] = "Player"


class ChatResponse(BaseModel):
    npc_id: str
    npc_name: str
    npc_response: str
    status: str = "success"


class NPCStatusResponse(BaseModel):
    npc_id: str
    status: dict


class CreateNPCRequest(BaseModel):
    npc_id: str
    name: str
    persona: str


# FastAPI 앱 초기화
app = FastAPI(title="NPC Chat Server", version="1.0.0")

# CORS 설정 (Unity에서 접근할 수 있도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
npc_agents = {}  # NPC ID -> NpcAgent 매핑
llm_utils = None
agent_lock = Lock()  # 스레드 안전성을 위한 락


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global llm_utils

    # API 키 확인
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk":
        raise Exception("OpenAI API 키를 config.py에 설정해주세요!")

    # LLM 유틸리티 초기화
    llm_utils = LLM_Utils(api_key=OPENAI_API_KEY)

    # 기본 NPC 생성 (이서아)
    default_npc = NpcAgent(
        name="이서아",
        persona="21살의 대학생. 시각 디자인을 전공하며 졸업 작품으로 고민이 많다.",
        llm_utils=llm_utils
    )
    npc_agents["seoa"] = default_npc

    print("✅ NPC 채팅 서버가 시작되었습니다!")
    print(f"✅ 기본 NPC '이서아' 생성됨 (ID: seoa)")


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "message": "NPC Chat Server is running!",
        "active_npcs": list(npc_agents.keys()),
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_with_npc(request: ChatRequest):
    """NPC와 대화하기"""
    try:
        with agent_lock:
            # NPC 존재 확인
            if request.npc_id not in npc_agents:
                raise HTTPException(status_code=404, detail=f"NPC '{request.npc_id}'를 찾을 수 없습니다.")

            # NPC 응답 생성
            npc = npc_agents[request.npc_id]
            npc_response = npc.respond_to_player(request.player_message)

            return ChatResponse(
                npc_id=request.npc_id,
                npc_name=npc.name,
                npc_response=npc_response
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 처리 중 오류: {str(e)}")


@app.post("/npc/create")
async def create_npc(request: CreateNPCRequest):
    """새로운 NPC 생성"""
    try:
        with agent_lock:
            if request.npc_id in npc_agents:
                raise HTTPException(status_code=400, detail=f"NPC '{request.npc_id}'는 이미 존재합니다.")

            # 새 NPC 생성
            new_npc = NpcAgent(
                name=request.name,
                persona=request.persona,
                llm_utils=llm_utils
            )
            npc_agents[request.npc_id] = new_npc

            return {
                "status": "success",
                "message": f"NPC '{request.name}' (ID: {request.npc_id}) 생성 완료",
                "npc_id": request.npc_id
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NPC 생성 중 오류: {str(e)}")


@app.get("/npc/{npc_id}/status", response_model=NPCStatusResponse)
async def get_npc_status(npc_id: str):
    """NPC 상태 조회"""
    try:
        with agent_lock:
            if npc_id not in npc_agents:
                raise HTTPException(status_code=404, detail=f"NPC '{npc_id}'를 찾을 수 없습니다.")

            npc = npc_agents[npc_id]
            status = npc.get_status()

            return NPCStatusResponse(
                npc_id=npc_id,
                status=status
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 조회 중 오류: {str(e)}")


@app.get("/npc/list")
async def list_npcs():
    """활성화된 NPC 목록 조회"""
    with agent_lock:
        npc_list = []
        for npc_id, npc in npc_agents.items():
            npc_list.append({
                "npc_id": npc_id,
                "name": npc.name,
                "persona": npc.persona[:50] + "..." if len(npc.persona) > 50 else npc.persona
            })

        return {
            "npcs": npc_list,
            "total_count": len(npc_list)
        }


@app.delete("/npc/{npc_id}")
async def delete_npc(npc_id: str):
    """NPC 삭제"""
    try:
        with agent_lock:
            if npc_id not in npc_agents:
                raise HTTPException(status_code=404, detail=f"NPC '{npc_id}'를 찾을 수 없습니다.")

            # 기본 NPC는 삭제 방지
            if npc_id == "seoa":
                raise HTTPException(status_code=400, detail="기본 NPC는 삭제할 수 없습니다.")

            del npc_agents[npc_id]

            return {
                "status": "success",
                "message": f"NPC '{npc_id}' 삭제 완료"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NPC 삭제 중 오류: {str(e)}")


# 서버 실행 함수
def run_server(host: str = "localhost", port: int = 8000):
    """서버 실행"""
    print(f"🚀 NPC 채팅 서버를 {host}:{port}에서 시작합니다...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()