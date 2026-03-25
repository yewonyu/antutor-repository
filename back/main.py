import asyncio
import json
import httpx
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from datetime import datetime, timedelta
import jwt
import bcrypt
from google import genai

# Import definitions and configuration from config.py
from config import (
    TARGET_CONCEPTS, CONCEPT_DICTIONARY, AZURE_PHI4_ENDPOINT, AZURE_PHI4_API_KEY, 
    PROMPTS, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    GIVE_UP_KEYWORDS, NEWS_API_KEY, GEMINI_API_KEY
)

gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Antutor Metric AI Backend", description="Sejong University Capstone Backend")

# ---------------------------------------------------------
# 1. AI 모델 초기화 (Model Initialization)
# ---------------------------------------------------------
print("Loading NLI model. This might take a minute on first run...")
try:
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
except Exception as e:
    print(f"Warning: Failed to load NLI DeBERTa model: {e}")
    nli_model = None

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

async def extract_atomic_propositions(user_answer: str, ground_truth: str) -> List[str]:
    # 1. API 키가 없으면 예외 발생 (에러 트래킹 목적)
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini API Key is not configured for Atomic Extraction.")
        
    # 2. 진짜 Gemini API 추출 로직 설계
    prompt = PROMPTS["atomic_extraction"].format(user_answer=user_answer, ground_truth=ground_truth)
    
    try:
        response = await gemini_client.aio.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        raw_text = response.text.strip()
        
        # 모델이 마크다운 블록(```json 등)을 반환할 수 있으므로 제거
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        return data.get("propositions", [user_answer])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API extraction failed: {str(e)}")

async def verify_nli_contradiction(user_answer: str, ground_truth: str) -> bool:
    if not nli_model:
        raise HTTPException(status_code=500, detail="NLI DeBERTa model is not loaded.")
    pairs = [[ground_truth, user_answer]]
    scores = await asyncio.to_thread(nli_model.predict, pairs)
    if np.argmax(scores[0]) == 0:
        return True
    return False

async def verify_nli_atomic(propositions: List[str], ground_truth: str) -> float:
    if not nli_model or not propositions:
        raise HTTPException(status_code=500, detail="NLI DeBERTa model is not loaded or propositions are empty.")
    
    entailment_count: int = 0
    pairs = [[ground_truth, prop] for prop in propositions]
    scores = await asyncio.to_thread(nli_model.predict, pairs)
    
    for score in scores:
        if np.argmax(score) == 1:
            entailment_count += 1
            
    return float(entailment_count) / len(propositions)

async def calculate_antutor_score(user_answer: str, ground_truth: str) -> tuple:
    propositions = await extract_atomic_propositions(user_answer, ground_truth)
    score = await verify_nli_atomic(propositions, ground_truth)
    return propositions, score

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

async def call_expert_agent(persona: str, concept: str, user_answer: str, context: Optional[str] = None, nli_score: Optional[float] = None) -> Dict[str, Any]:
    if persona == "The Academic Auditor":
        prompt = PROMPTS["experts"][persona].format(concept=concept, user_answer=user_answer, nli_score=nli_score)
    else:
        prompt = PROMPTS["experts"][persona].format(concept=concept, user_answer=user_answer, context=context)

    # Azure OpenAI/MaaS Chat Completions 포맷에 맞춘 페이로드
    payload = {
        "model": "Phi-4", # Azure AI Studio의 배포 이름(Deployment Name)과 동일하게 맞춰야 함
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_PHI4_API_KEY}"
    }
    
    # Azure API 호출
    async with httpx.AsyncClient() as client:
        try:
            if not AZURE_PHI4_ENDPOINT or not AZURE_PHI4_API_KEY:
                raise HTTPException(status_code=500, detail="Azure Phi-4 Endpoint or API Key is missing in .env configuration.")

            response = await client.post(AZURE_PHI4_ENDPOINT, headers=headers, json=payload, timeout=120.0)
            
            # API 권한 오류나 기타 상태 에러 발생 시 예외 생성
            response.raise_for_status()
            
            result = response.json()
            feedback = result["choices"][0]["message"]["content"].strip()
                
            score = None
            if persona != "The Academic Auditor":
                import re
                match = re.search(r'\[\s*(0\.\d+|1\.00?)\s*\]', feedback)
                if match:
                    try:
                        score = float(match.group(1))
                        feedback = re.sub(r'\[\s*(0\.\d+|1\.00?)\s*\]', '', feedback).strip()
                    except ValueError:
                        pass
                
            return {"persona": persona, "feedback": feedback, "score": score}
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail=f"[{persona} Timeout] Azure server timed out after 120 seconds.")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=500, detail=f"[{persona} Azure API Error] Status {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"[{persona} Connection Error] Failed to generate feedback: {str(e)}")

async def generate_moderator_guidance_message(user_answer: str, lowest_persona: str, expert_results: List[Dict]) -> str:
    if not gemini_client:
        return f"Your answer is a good start, but needs more depth from the perspective of {lowest_persona}. Try to expand on that!"
        
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
    try:
        response = await gemini_client.aio.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Your answer needs improvement, particularly in the perspective of {lowest_persona}. Please try again considering their feedback!"

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
        # 정답과 완전히 모순되는지 검증
        is_contradiction = await verify_nli_contradiction(request.user_answer, ground_truth)
        if is_contradiction:
            # 모순이 발생하면 오답 처리 및 점수 0점 부여
            antutor_score = 0.0
            propositions = ["NLI Blocked: Explicit contradiction found."]
            expert_results = [{"persona": "System", "score": 0.0, "feedback": "Answer contradicts the ground truth."}]
            expert_scores = {"System": 0.0, "The Market Practitioner": 0.0, "The Macro-Connector": 0.0, "The Academic Auditor": 0.0}
        else:
            # 2 & 3. Atomic-NLI(명제 추출 및 정렬 점수 산출) 평가와 외부 컨텍스트(뉴스, 지식 그래프) 병렬 수집
            (propositions, antutor_score), news_context, kg_context = await asyncio.gather(
                calculate_antutor_score(request.user_answer, ground_truth),
                retrieve_news_rag(concept),
                retrieve_knowledge_graph(concept)
            )

            # 4. 세 명의 전문가 에이전트 지정 (Academic -> Market -> Macro 순서)
            personas = ["The Academic Auditor", "The Market Practitioner", "The Macro-Connector"]
            # 5. 수집된 컨텍스트와 Antutor 스코어를 넣어 각각의 에이전트들에게 비동기로 피드백 생성을 요청
            # Market과 Macro는 자체적으로 프롬프트를 통해 점수 생성
            tasks = [
                call_expert_agent("The Academic Auditor", concept, request.user_answer, nli_score=antutor_score),
                call_expert_agent("The Market Practitioner", concept, request.user_answer, context=news_context),
                call_expert_agent("The Macro-Connector", concept, request.user_answer, context=kg_context)
            ]
            
            # 병렬로 던진 모든 에이전트의 피드백이 완성될 때까지 대기
            expert_results = list(await asyncio.gather(*tasks))

            # 6. 각 전문가별 점수 할당 로직
            # Academic Auditor는 NLI-atomic 알고리즘으로 계산된 antutor_score를 그대로 사용
            expert_scores_raw = {"The Academic Auditor": antutor_score}
            for res in expert_results:
                if res["persona"] == "The Academic Auditor":
                    res["score"] = antutor_score
                else:
                    # 에이전트가 문자열 내에서 추출한 점수를 사용. 파싱 실패 시 기본값 0.75
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
