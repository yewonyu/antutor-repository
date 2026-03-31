import os
from dotenv import load_dotenv

# .env 파일의 환경 변수를 불러옵니다.
load_dotenv()
# --- Authentication Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY", "your-very-secure-secret-key-capstone-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week expiration

TARGET_CONCEPTS = {
    "Inflation": {
        "definition": "The rate at which the general level of prices for goods and services is rising and, consequently, the purchasing power of currency is falling.",
        "initial_question": "How would you explain inflation and its impact on everyday purchasing power?",
        "sub_concept_name": "Purchasing Power",
        "sub_concept_question": "It seems you're struggling with the main concept. Before we talk about inflation, can you first explain what 'Purchasing Power' means?",
        "dictionary_link": "/dictionary/Inflation"
    },
    "Interest Rate": {
        "definition": "The proportion of a loan that is charged as interest to the borrower, typically expressed as an annual percentage of the loan outstanding.",
        "initial_question": "What is an interest rate, and how does it affect both borrowers and savers?",
        "sub_concept_name": "Principal and Percentage",
        "sub_concept_question": "Let's step back. Can you explain what a 'Principal' loan amount is and how a 'Percentage' applies to it?",
        "dictionary_link": "/dictionary/InterestRate"
    },
    "Exchange Rate": {
        "definition": "The value of one currency for the purpose of conversion to another.",
        "initial_question": "Can you describe what an exchange rate is and why it fluctuates?",
        "sub_concept_name": "Currency Pairing",
        "sub_concept_question": "Let's simplify. Why do countries have different currencies, and why do we need to swap them?",
        "dictionary_link": "/dictionary/ExchangeRate"
    },
    "Opportunity Cost": {
        "definition": "The loss of potential gain from other alternatives when one alternative is chosen.",
        "initial_question": "Explain the concept of opportunity cost with a real-life example.",
        "sub_concept_name": "Trade-offs",
        "sub_concept_question": "Before defining opportunity cost, can you provide an example of a simple 'trade-off' in your everyday life?",
        "dictionary_link": "/dictionary/OpportunityCost"
    },
    "Compound Interest": {
        "definition": "Interest calculated on the initial principal and also on the accumulated interest of previous periods of a deposit or loan.",
        "initial_question": "Why is compound interest often called the eighth wonder of the world? How does it work?",
        "sub_concept_name": "Simple Interest",
        "sub_concept_question": "To understand compound interest, could you first explain how 'Simple Interest' works?",
        "dictionary_link": "/dictionary/CompoundInterest"
    }
}

CONCEPT_DICTIONARY = {
    "Inflation": {
        "term": "Inflation",
        "simple_definition": "Inflation means that things get more expensive over time, so your money buys less than it used to.",
        "example": "If an apple costs $1 this year and $1.05 next year due to inflation, your $1 bill can no longer buy that apple."
    },
    "InterestRate": {
        "term": "Interest Rate",
        "simple_definition": "An interest rate is the extra money you pay when you borrow money, or the extra money you earn when you save money in a bank.",
        "example": "If you borrow $100 at a 5% interest rate, you will have to pay back $105."
    },
    "ExchangeRate": {
        "term": "Exchange Rate",
        "simple_definition": "An exchange rate is the price of one country's money in terms of another country's money.",
        "example": "If the exchange rate for USD to KRW is 1,300, it means 1 US Dollar can be traded for 1,300 Korean Won."
    },
    "OpportunityCost": {
        "term": "Opportunity Cost",
        "simple_definition": "Opportunity cost is what you give up when you choose to do one thing over another.",
        "example": "If you spend Friday night studying instead of going to a movie, the opportunity cost is the fun you missed at the movie."
    },
    "CompoundInterest": {
        "term": "Compound Interest",
        "simple_definition": "Compound interest is 'interest on interest'. You earn money not just on the original amount, but also on the interest that has already piled up.",
        "example": "If you have $100 and it earns 10% compound interest, year 1 you get $10. Year 2 you earn 10% on $110, so you get $11, and it grows faster and faster."
    }
}

LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:11434/api/chat")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma3:12b")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

PROMPTS = {
    "experts": {
        "The Academic Auditor": """You are 'The Academic Auditor'. You act as an impartial evaluator to assess a student's answer against the formal ground truth.
Concept: {concept}
Ground Truth: {ground_truth}
User Answer: {user_answer}

You MUST return a valid JSON object with exactly three keys:
1. "is_contradiction": boolean (true if the user's answer explicitly contradicts the fundamental meaning of the ground truth, false otherwise)
2. "score": float from 0.0 to 1.0 representing accuracy.
3. "feedback": string containing a brief, encouraging assessment.

Return ONLY the JSON object, with no markdown tags (e.g., no ```json).""",
        "The Market Practitioner": "You are 'The Market Practitioner'. Evaluate the concept explanation based on real-world market impacts. Incorporate the following News API RAG context into your assessment.\nNews Context: {context}\nConcept: {concept}\nUser Answer: {user_answer}\nProvide a brief assessment. Finally, on a new line, provide a numerical score from 0.00 to 1.00 enclosed in brackets, e.g., [0.85].",
        "The Macro-Connector": "You are 'The Macro-Connector'. Evaluate how well the explanation connects the concept to broader macroeconomic trends. Incorporate the following Knowledge Graph context into your assessment.\nKnowledge Graph Context: {context}\nConcept: {concept}\nUser Answer: {user_answer}\nProvide a brief assessment. Finally, on a new line, provide a numerical score from 0.00 to 1.00 enclosed in brackets, e.g., [0.85]."
    }
}

# --- Keyword Detection for Give-Up ---
GIVE_UP_KEYWORDS = [
    "don't know", "give up", "not sure", "no idea", "hint", 
    "can't explain", "too hard", "stuck", "confused", "help"
]

