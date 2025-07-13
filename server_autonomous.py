# server_autonomous.py
import threading
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

from config_autonomous import *
from llm_utils import LLM_Utils
from npc_agent_autonomous import AutonomousNpcAgent
from time_manager import time_manager


# 요청/응답 모델들
class ChatRequest(BaseModel):
    npc_id: str
    player_message: str
    player_name: Optional[str] = "Player"
    player_location: Optional[str] = None


class ChatResponse(BaseModel):
    npc_id: str
    npc_name: str
    npc_response: str
    npc_status: Dict
    status: str = "success"


class NPCStatusResponse(BaseModel):
    npc_id: str
    status: Dict
    unity_commands: Optional[Dict] = None


class CreateNPCRequest(BaseModel):
    npc_id: str
    name: str
    persona: str


class TimeControlRequest(BaseModel):
    action: str  # "start", "stop", "set_speed"
    speed: Optional[float] = None


# FastAPI 앱 초기화
app = FastAPI(title="Autonomous NPC Server", version="2.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수들
npc_agents: Dict[str, AutonomousNpcAgent] = {}
llm_utils = None
autonomous_loop_running = False
autonomous_thread = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global llm_utils, autonomous_thread

    # API 키 확인
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-api-key-here":
        raise Exception("OpenAI API 키를 config_autonomous.py에 설정해주세요!")

    # LLM 유틸리티 초기화
    llm_utils = LLM_Utils(api_key=OPENAI_API_KEY)

    # 기본 NPC 생성 (이서아)
    default_npc = AutonomousNpcAgent(
        name="이서아",
        persona="21살의 대학생. 시각 디자인을 전공하며 졸업 작품으로 고민이 많다. 평소 도서관에서 공부하거나 카페에서 휴식하는 것을 좋아한다.",
        llm_utils=llm_utils
    )
    npc_agents["seoa"] = default_npc

    # 시간 관리자 시작
    time_manager.start_time_flow()

    # 자율 행동 시스템 시작
    start_autonomous_system()

    print("✅ 자율 NPC 서버가 시작되었습니다!")
    print(f"✅ 기본 NPC '이서아' 생성됨 (ID: seoa)")
    print(f"✅ 게임 시간: {time_manager.get_time_str()}")


def start_autonomous_system():
    """자율 행동 시스템 시작"""
    global autonomous_loop_running, autonomous_thread

    if not autonomous_loop_running:
        autonomous_loop_running = True
        autonomous_thread = threading.Thread(target=autonomous_update_loop, daemon=True)
        autonomous_thread.start()
        print("✅ 자율 행동 시스템 시작됨")


def autonomous_update_loop():
    """자율 행동 업데이트 루프 (백그라운드 스레드)"""
    global autonomous_loop_running

    while autonomous_loop_running:
        try:
            # 모든 NPC의 자율 행동 업데이트
            for npc_id, npc in npc_agents.items():
                if npc.is_autonomous_mode:
                    npc.autonomous_update()

            time.sleep(AUTONOMOUS_UPDATE_INTERVAL)

        except Exception as e:
            print(f"[ERROR] 자율 행동 업데이트 오류: {e}")
            time.sleep(5)


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "message": "Autonomous NPC Server is running!",
        "current_time": time_manager.get_time_str(),
        "active_npcs": list(npc_agents.keys()),
        "autonomous_system": autonomous_loop_running,
        "version": "2.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_with_npc(request: ChatRequest):
    """NPC와 대화하기 (자율 행동 버전)"""
    try:
        # NPC 존재 확인
        if request.npc_id not in npc_agents:
            raise HTTPException(status_code=404, detail=f"NPC '{request.npc_id}'를 찾을 수 없습니다.")

        # NPC 응답 생성
        npc = npc_agents[request.npc_id]
        npc_response = npc.respond_to_player(
            request.player_message,
            request.player_location
        )

        # NPC 상태 정보 가져오기
        npc_status = npc.get_status_for_unity()

        return ChatResponse(
            npc_id=request.npc_id,
            npc_name=npc.name,
            npc_response=npc_response,
            npc_status=npc_status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 처리 중 오류: {str(e)}")


@app.get("/npc/{npc_id}/status", response_model=NPCStatusResponse)
async def get_npc_status(npc_id: str):
    """NPC 상태 조회 (자율 행동 정보 포함)"""
    try:
        if npc_id not in npc_agents:
            raise HTTPException(status_code=404, detail=f"NPC '{npc_id}'를 찾을 수 없습니다.")

        npc = npc_agents[npc_id]
        status = npc.get_status_for_unity()
        unity_commands = status.get('movement_command')

        return NPCStatusResponse(
            npc_id=npc_id,
            status=status,
            unity_commands=unity_commands
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 조회 중 오류: {str(e)}")


@app.get("/npc/{npc_id}/debug")
async def get_npc_debug_info(npc_id: str):
    """NPC 디버그 정보 조회"""
    try:
        if npc_id not in npc_agents:
            raise HTTPException(status_code=404, detail=f"NPC '{npc_id}'를 찾을 수 없습니다.")

        npc = npc_agents[npc_id]
        debug_info = npc.get_debug_info()

        return debug_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"디버그 정보 조회 중 오류: {str(e)}")


@app.post("/npc/{npc_id}/end_interaction")
async def end_npc_interaction(npc_id: str):
    """NPC와의 상호작용 종료"""
    try:
        if npc_id not in npc_agents:
            raise HTTPException(status_code=404, detail=f"NPC '{npc_id}'를 찾을 수 없습니다.")

        npc = npc_agents[npc_id]
        npc.end_player_interaction()

        return {
            "status": "success",
            "message": f"NPC '{npc_id}'와의 상호작용이 종료되었습니다."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상호작용 종료 중 오류: {str(e)}")


@app.get("/time/status")
async def get_time_status():
    """게임 시간 상태 조회"""
    return {
        "current_time": time_manager.get_time_str(),
        "current_date": time_manager.get_curr_date_str(),
        "time_speed": time_manager.time_speed,
        "is_running": time_manager.is_running,
        "start_time": time_manager.start_time.strftime("%B %d, %Y, %H:%M:%S")
    }


@app.post("/time/control")
async def control_time(request: TimeControlRequest):
    """게임 시간 제어"""
    try:
        if request.action == "start":
            time_manager.start_time_flow()
            return {"status": "success", "message": "시간 흐름이 시작되었습니다."}

        elif request.action == "stop":
            time_manager.stop_time_flow()
            return {"status": "success", "message": "시간 흐름이 정지되었습니다."}

        elif request.action == "set_speed":
            if request.speed is None:
                raise HTTPException(status_code=400, detail="속도 값이 필요합니다.")
            time_manager.set_time_speed(request.speed)
            return {"status": "success", "message": f"시간 배속이 {request.speed}x로 설정되었습니다."}

        else:
            raise HTTPException(status_code=400, detail="유효하지 않은 액션입니다.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시간 제어 중 오류: {str(e)}")


@app.get("/npc/list")
async def list_npcs():
    """활성화된 NPC 목록 조회 (자율 행동 정보 포함)"""
    npc_list = []
    for npc_id, npc in npc_agents.items():
        status = npc.get_status_for_unity()
        npc_list.append({
            "npc_id": npc_id,
            "name": npc.name,
            "persona": npc.persona[:50] + "..." if len(npc.persona) > 50 else npc.persona,
            "current_action": status.get('current_action', '알 수 없음'),
            "location": status.get('location', '알 수 없음'),
            "emotion": status.get('emotion', '평온함'),
            "autonomous_mode": npc.is_autonomous_mode,
            "interaction_available": status.get('interaction_available', True)
        })

    return {
        "npcs": npc_list,
        "total_count": len(npc_list),
        "current_time": time_manager.get_time_str()
    }


@app.post("/npc/create")
async def create_npc(request: CreateNPCRequest):
    """새로운 자율 NPC 생성"""
    try:
        if request.npc_id in npc_agents:
            raise HTTPException(status_code=400, detail=f"NPC '{request.npc_id}'는 이미 존재합니다.")

        # 새 NPC 생성
        new_npc = AutonomousNpcAgent(
            name=request.name,
            persona=request.persona,
            llm_utils=llm_utils
        )
        npc_agents[request.npc_id] = new_npc

        return {
            "status": "success",
            "message": f"자율 NPC '{request.name}' (ID: {request.npc_id}) 생성 완료",
            "npc_id": request.npc_id,
            "initial_status": new_npc.get_status_for_unity()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NPC 생성 중 오류: {str(e)}")


@app.delete("/npc/{npc_id}")
async def delete_npc(npc_id: str):
    """NPC 삭제"""
    try:
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


@app.get("/system/stats")
async def get_system_stats():
    """시스템 통계 정보"""
    total_memories = sum(
        len(npc.memory_manager.seq_event) + len(npc.memory_manager.seq_thought)
        for npc in npc_agents.values()
    )

    total_knowledge = sum(
        len(npc.memory_manager.knowledge_base)
        for npc in npc_agents.values()
    )

    return {
        "total_npcs": len(npc_agents),
        "total_memories": total_memories,
        "total_knowledge": total_knowledge,
        "autonomous_system_running": autonomous_loop_running,
        "time_system_running": time_manager.is_running,
        "current_game_time": time_manager.get_time_str(),
        "time_speed": time_manager.time_speed,
        "uptime_info": {
            "server_start": "서버 시작 시간 정보",  # 실제로는 시작 시간 추적 구현
            "game_days_passed": "게임 내 경과 일수"  # 실제로는 경과 일수 계산 구현
        }
    }


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    global autonomous_loop_running

    print("🔄 서버 종료 중...")

    # 자율 행동 시스템 정지
    autonomous_loop_running = False

    # 시간 관리자 정지
    time_manager.stop_time_flow()

    print("✅ 서버가 안전하게 종료되었습니다.")


# 서버 실행 함수
def run_server(host: str = "localhost", port: int = 8000):
    """서버 실행"""
    import uvicorn
    print(f"🚀 자율 NPC 서버를 {host}:{port}에서 시작합니다...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()