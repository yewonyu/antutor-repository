import asyncio
import json
import httpx
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
import bcrypt

# Import definitions and configuration from config.py
from config import (
    TARGET_CONCEPTS, CONCEPT_DICTIONARY, LOCAL_LLM_ENDPOINT, LOCAL_LLM_MODEL, 
    PROMPTS, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    GIVE_UP_KEYWORDS, NEWS_API_KEY
)

app = FastAPI(title="Antutor Metric AI Backend", description="Sejong University Capstone Backend")

# 이 코드가 있어야 프론트엔드에서 백엔드 데이터를 읽을 수 있음
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # 리액트 주소
    allow_credentials=True,
    allow_methods=["*"], # 모든 방식(GET, POST 등) 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# ---------------------------------------------------------
# 1. AI 모델 초기화 (Model Initialization)
# ---------------------------------------------------------
# NLI model usage has been removed in favor of a local LLM-as-a-judge.

# In-Memory DBs
session_memory: Dict[str, Any] = {}
users_db: Dict[str, Any] = {}

# ---------------------------------------------------------
# 2. Schema Definitions
# ---------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    session_id: str
    concept: str
    user_answer: str

class EndSessionRequest(BaseModel):
    session_id: str

# ---------------------------------------------------------
# 3. 인증 및 권한 처리 (Authentication & Authorization)
# ---------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    if username not in users_db:
        raise credentials_exception
    return username

@app.get("/check-username")
async def check_username(username: str):
    """지정된 아이디(username)가 이미 데이터베이스에 존재하는지 실시간으로 확인합니다."""
    if username in users_db:
        return {"available": False, "message": "Username already exists."}
    return {"available": True, "message": "Username is available."}

@app.post("/register")
async def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    users_db[user.username] = {
        "hashed_password": get_password_hash(user.password),
        "history": {},
        "completed_concepts": []
    }
    return {"message": "User successfully registered"}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ---------------------------------------------------------
# 4. 핵심 로직: 하이브리드 게이트키퍼 및 Antutor Metric (Core Logic)
# ---------------------------------------------------------

async def call_local_llm(prompt: str, is_json: bool = False, model: Optional[str] = None) -> str:
    """Helper to call local LLM API (e.g., Ollama)"""
    model_name = model or LOCAL_LLM_MODEL
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    if is_json:
        payload["format"] = "json"
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(LOCAL_LLM_ENDPOINT, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
            
            if "message" in data:
                return data["message"]["content"]
            elif "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
    except Exception as e:
        print(f"Local LLM Call Error: {str(e)}")
        if is_json:
            return '{"is_contradiction": false, "score": 0.5, "feedback": "API error."}'
    
    return "Local LLM Error."

async def evaluate_academic_auditor(concept: str, user_answer: str, ground_truth: str) -> dict:
    prompt = PROMPTS["experts"]["The Academic Auditor"].format(
        concept=concept, ground_truth=ground_truth, user_answer=user_answer
    )
    
    raw_response = await call_local_llm(prompt, is_json=True)
    try:
        return json.loads(raw_response)
    except Exception:
        return {
            "is_contradiction": False,
            "score": 0.5,
            "feedback": "Failed to parse local LLM assessment."
        }

async def retrieve_news_rag(concept: str) -> str:
    # 1. NEWS_API_KEY가 없는 경우 예외 발생
    if not NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="NEWS_API_KEY is not configured.")
    
    # 2. News API 실제 호출 로직
    url = f"https://newsapi.org/v2/everything?q={concept}&sortBy=relevancy&pageSize=3&apiKey={NEWS_API_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            data = response.json()
            if data.get("status") == "ok" and data.get("articles"):
                articles = data["articles"]
                news_summary = " ".join([f"Headline: {art['title']}." for art in articles])
                return f"Recent news context for {concept}: {news_summary}"
            else:
                return f"No recent news found for {concept}."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching news for {concept}: {str(e)}")
    
    raise HTTPException(status_code=500, detail=f"Failed to retrieve news context for {concept}.")

async def retrieve_knowledge_graph(concept: str) -> str:
    # Simulating a Knowledge Graph retrieval
    await asyncio.sleep(0.5)
    return f"Knowledge Graph Node [{concept}] -> Connected to [Global Trade, Employment Rates, Inflation]. Policy changes directly impact these connected nodes."

async def call_expert_agent(persona: str, concept: str, user_answer: str, context: Optional[str] = None) -> Dict[str, Any]:
    prompt = PROMPTS["experts"][persona].format(concept=concept, user_answer=user_answer, context=context)
    
    try:
        feedback = await call_local_llm(prompt, is_json=False)
        score = None
        import re
        match = re.search(r'\[\s*(0\.\d+|1\.00?)\s*\]', feedback)
        if match:
            try:
                score = float(match.group(1))
                feedback = re.sub(r'\[\s*(0\.\d+|1\.00?)\s*\]', '', feedback).strip()
            except ValueError:
                pass
                
        return {"persona": persona, "feedback": feedback, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[{persona} Connection Error] Failed to generate feedback: {str(e)}")

async def generate_moderator_guidance_message(user_answer: str, lowest_persona: str, expert_results: List[Dict]) -> str:
    lowest_feedback = next((res["feedback"] for res in expert_results if res["persona"] == lowest_persona), "")
    
    prompt = f"""
You are the friendly Lead Tutor guiding a student. 
The student provided this answer: "{user_answer}"
The expert '{lowest_persona}' evaluated the answer and gave this feedback: "{lowest_feedback}"

Write a short, encouraging message in English (1-3 sentences) directly replying to the student. 
1. Point out exactly what they missed based ONLY on the {lowest_persona}'s feedback.
2. End with a follow-up question to help them think about that missing aspect.
Do NOT give them the direct answer.
"""
    return await call_local_llm(prompt, is_json=False)

# ---------------------------------------------------------
# 5. Application API Endpoints
# ---------------------------------------------------------

@app.get("/dictionary")
async def get_all_dictionary_terms():
    return list(CONCEPT_DICTIONARY.keys())

@app.get("/dictionary/{term}")
async def get_dictionary_term(term: str):
    if term not in CONCEPT_DICTIONARY:
        raise HTTPException(status_code=404, detail="Term not found in dictionary.")
    return CONCEPT_DICTIONARY[term]

@app.get("/start/{concept}")
async def start_session(concept: str, current_user: str = Depends(get_current_user)):
    if concept not in TARGET_CONCEPTS:
        raise HTTPException(status_code=404, detail="Target Concept is not supported.")
    
    import uuid
    session_id = str(uuid.uuid4())
    
    session_memory[session_id] = {
        "user_id": current_user,
        "concept": concept,
        "scaffold_level": 0,  
        "scaffold_count": 0,  # Tracks how many times the user gave up
        "history": [],
        "radar_data": {"Academic": [], "Market": [], "Macro": []}
    }
    
    return {
        "session_id": session_id,
        "concept": concept,
        "initial_question": TARGET_CONCEPTS[concept]["initial_question"]
    }

@app.post("/chat")
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    if request.session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Invalid Session ID.")
        
    session = session_memory[request.session_id]
    if session["user_id"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to access this session.")
        
    concept = request.concept
    ground_truth = TARGET_CONCEPTS[concept]["definition"]
    
    # 포기 키워드가 입력되었는지 확인 (Check Give-up keywords)
    is_give_up = any(kw in request.user_answer.lower() for kw in GIVE_UP_KEYWORDS)
    is_contradiction = False
    
    # 사용자 답변 평가 단계 (Evaluate Answer)
    if is_give_up:
        # 사용자가 포기한 경우, 모든 점수를 0으로 처리하고 도움을 제공
        antutor_score = 0.0
        expert_scores = {"The Market Practitioner": 0.0, "The Macro-Connector": 0.0, "The Academic Auditor": 0.0}
        propositions = []
        expert_results = [{"persona": "System", "score": 0.0, "feedback": "User requested help."}]
        lowest_persona = "System"
    else:
        # LLM-as-a-judge (단일 평가 방식으로 변경)
        academic_result = await evaluate_academic_auditor(concept, request.user_answer, ground_truth)
        is_contradiction = academic_result.get("is_contradiction", False)
        antutor_score = academic_result.get("score", 0.0)
        
        if is_contradiction:
            # 모순이 발생하면 오답 처리 및 점수 0점 부여
            antutor_score = 0.0
            propositions = ["Local LLM Blocked: Explicit contradiction found."]
            expert_results = [{"persona": "System", "score": 0.0, "feedback": "Answer contradicts the ground truth."}]
            expert_scores = {"System": 0.0, "The Market Practitioner": 0.0, "The Macro-Connector": 0.0, "The Academic Auditor": 0.0}
        else:
            propositions = ["(Atomic extraction skipped: Evaluated by LLM-as-a-judge in one go)"]
            
            # 외부 컨텍스트 병렬 수집 (News, KG)
            news_context, kg_context = await asyncio.gather(
                retrieve_news_rag(concept),
                retrieve_knowledge_graph(concept)
            )

            # 나머지 두 전문가 (Market, Macro) 에이전트 평가 병렬 요청
            tasks = [
                call_expert_agent("The Market Practitioner", concept, request.user_answer, context=news_context),
                call_expert_agent("The Macro-Connector", concept, request.user_answer, context=kg_context)
            ]
            
            other_expert_results = list(await asyncio.gather(*tasks))

            # 합산 결과 리스트 구성
            expert_results = [
                {"persona": "The Academic Auditor", "score": antutor_score, "feedback": academic_result.get("feedback", "")}
            ] + other_expert_results

            # 각 전문가별 점수 할당 로직
            expert_scores_raw = {"The Academic Auditor": antutor_score}
            for res in other_expert_results:
                res["score"] = res.get("score") if res.get("score") is not None else 0.75
                expert_scores_raw[res["persona"]] = res["score"]
                    
            expert_scores = expert_scores_raw
            
        # 7. 세 명 중 가장 점수가 낮은(취약한) 전문가 영역을 찾아 모더레이터(Scaffolding) 판단 기준으로 사용
        lowest_persona = min(expert_scores.keys(), key=lambda k: expert_scores[k])
        
    # 전문가 3인의 평균 점수 계산 (Expert Average)
    raw_avg_score = (
        expert_scores.get("The Academic Auditor", 0) * 100 +
        expert_scores.get("The Market Practitioner", 0) * 100 +
        expert_scores.get("The Macro-Connector", 0) * 100
    ) / 3.0

    # 모더레이터 판단 및 스캐폴딩(학습 보조) 로직 (Moderator & Scaffolding Logic)
    moderator_action = "proceed"
    scaffold_plan = None
    current_scaffold_level = session["scaffold_level"]

    if is_give_up:
        # 사용자가 포기한 경우 스캐폴딩(점진적 힌트) 제공
        session["scaffold_count"] += 1
        moderator_action = "scaffold"
        if current_scaffold_level == 0:
            session["scaffold_level"] = 1
            scaffold_plan = {
                "step": "Sub-concept Nudge",
                "message": TARGET_CONCEPTS[concept]["sub_concept_question"]
            }
        elif current_scaffold_level >= 1:
            session["scaffold_level"] = 2
            term_key = TARGET_CONCEPTS[concept]["dictionary_link"].split("/")[-1]
            dict_info = CONCEPT_DICTIONARY.get(term_key, {})
            scaffold_plan = {
                "step": "Concept Dictionary Link",
                "message": "It looks like you need help. Here is the concept dictionary link.",
                "dictionary_link": TARGET_CONCEPTS[concept]["dictionary_link"],
                "definition": dict_info.get("simple_definition", "")
            }
    else:
        # 포기하지 않은 경우 평균 점수를 확인 (Check average)
        if raw_avg_score >= 85:
            # 평균 85점 이상이면 세션 종료 제안
            moderator_action = "suggest_termination"
            scaffold_plan = {
                "step": "Termination Suggestion",
                "message": "You have achieved a high level of mastery. Would you like to terminate the session? (Yes/No)"
            }
        elif is_contradiction:
            # 명시적인 모순이 있으면 기존처럼 간단한 재시도 권유
            moderator_action = "retry"
            scaffold_plan = {
                "step": "Retry Prompt",
                "message": "Your answer seems to contradict the core facts. Please review the concept once more or ask for a 'hint'!"
            }
        else:
            # 정답에 근접했지만 아직 평균 85점에 도달하지 못한 경우 혹은 점수가 낮지만 모순은 없는 경우
            # 가장 점수가 낮은 영역을 보완할 수 있도록 가이드 제공
            moderator_action = "proceed"
            guidance_message = await generate_moderator_guidance_message(request.user_answer, lowest_persona, expert_results)
            
            scaffold_plan = {
                "step": "Guidance Prompt",
                "message": guidance_message
            }

    # 분석용 데이터 업데이트 (Update Analytics Data)
    session["radar_data"]["Academic"].append(antutor_score * 100)
    session["radar_data"]["Market"].append(expert_scores.get("The Market Practitioner", 0) * 100)
    session["radar_data"]["Macro"].append(expert_scores.get("The Macro-Connector", 0) * 100)
    
    session["history"].append({
        "user_answer": request.user_answer,
        "nli_score": antutor_score,
        "action": moderator_action
    })

    return {
        "atomic_propositions": propositions,
        "expert_average_score": raw_avg_score,
        "is_contradiction_override": is_contradiction,
        "expert_feedback": expert_results,
        "moderator_decision": {
            "status": moderator_action,
            "lowest_performing_area": lowest_persona,
            "scaffold_plan": scaffold_plan
        }
    }

@app.post("/end_session")
async def end_session(request: EndSessionRequest, current_user: str = Depends(get_current_user)):
    """
    Terminates the learning session, calculates final score including bonuses,
    and returns educational insights and growth visualization.
    """
    if request.session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Invalid Session ID.")
        
    session = session_memory[request.session_id]
    if session["user_id"] != current_user:
        raise HTTPException(status_code=403, detail="Not authorized to access this session.")
        
    nudge_count = session.get("scaffold_count", 0)
    academic_scores = session["radar_data"]["Academic"]
    market_scores = session["radar_data"]["Market"]
    macro_scores = session["radar_data"]["Macro"]
    
    last_academic = academic_scores[-1] if academic_scores else 0
    last_market = market_scores[-1] if market_scores else 0
    last_macro = macro_scores[-1] if macro_scores else 0
    
    latest_avg = (last_academic + last_market + last_macro) / 3.0
    
    if nudge_count == 0:
        final_score = latest_avg * 1.5
        educational_insights = f"Excellent! Your base average was {latest_avg:.1f}. You earned a 1.5x bonus for completing without help, making your final score {final_score:.1f}!"
    else:
        final_score = latest_avg
        educational_insights = f"Your score is {latest_avg:.1f}. You received help from the agent {nudge_count} times. Try harder next time for a bonus score!"
        
    # Check First Time
    user_data = users_db[current_user]
    concept = session["concept"]
    completed = user_data.get("completed_concepts", [])
    
    is_first_time = concept not in completed
    if is_first_time:
        user_data["completed_concepts"].append(concept)
        
    # Free up memory (cleanup in real DBs)
    radar_payload = session["radar_data"]
    
    return {
        "message": "Session terminated successfully.",
        "final_score": final_score,
        "educational_insights": educational_insights,
        "is_first_time": is_first_time,
        "growth_visualization": radar_payload
    }
