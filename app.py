import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

print("Initializing Gemini-Powered Enterprise Debt Recovery System...\n")

# --- 1. SET YOUR API KEY ---
# In a real app, never hardcode this. Use a .env file.
os.environ["GOOGLE_API_KEY"] = "AIzaSyCNUpDxv2UTO6MexX7wFdX3Qs3B_g4vxR8"

# --- 2. THE QUANTITATIVE TOOL (Your Mathematical Edge) ---
def predict_liquidity_and_ptp(debtor_data_str: str) -> str:
    """
    Simulates a predictive model to forecast Propensity to Pay (PTP).
    """
    try:
        data = eval(debtor_data_str)
        debt_amount = data.get("debt", 0)
        days_late = data.get("days_late", 0)
    except:
        return "Error parsing debtor data. Please provide as a dictionary."

    # Simulated output based on the math
    base_ptp = max(0.1, 1.0 - (days_late / 365.0))
    optimal_settlement = debt_amount * 0.60 if base_ptp < 0.4 else debt_amount * 0.85
    
    return f"ANALYSIS COMPLETE: Propensity to Pay is {base_ptp*100:.1f}%. Recommended settlement offer is ${optimal_settlement:.2f}."

quant_tool = Tool(
    name="Predictive_Financial_Model",
    func=predict_liquidity_and_ptp,
    description="Calculates debtor propensity to pay. Input must be a dictionary string like: {'debt': 5000, 'days_late': 120}"
)

# --- 3. THE GEMINI NLP CHAIN (The Negotiator) ---
# We initialize the Gemini 1.5 Pro model here
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

negotiation_prompt = PromptTemplate(
    input_variables=["financial_analysis", "debtor_message"],
    template="""
    You are an expert, compliant debt collection negotiator. 
    A debtor sent this message: "{debtor_message}"
    
    The Quantitative Analyst Agent provided this data: {financial_analysis}
    
    Draft a professional, empathetic, but firm 3-sentence email responding to the debtor and offering the settlement amount suggested by the Analyst.
    """
)
negotiation_chain = LLMChain(llm=llm, prompt=negotiation_prompt)

# --- 4. THE SUPERVISOR AGENT (The Orchestrator) ---
tools = [quant_tool]

supervisor_agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True # Shows the Gemini reasoning process
)

# --- 5. EXECUTE THE PIPELINE ---
def process_debtor_account(debtor_dict, debtor_message):
    print("\n==================================================")
    print(f"NEW FILE RECEIVED. DEBT: ${debtor_dict['debt']} | MESSAGE: '{debtor_message}'")
    print("==================================================\n")
    
    print(">>> GEMINI SUPERVISOR THOUGHT PROCESS:")
    financial_analysis = supervisor_agent.run(
        f"Use the Predictive_Financial_Model to analyze this debtor: {debtor_dict}."
    )
    
    print("\n>>> DRAFTING NLP RESPONSE:")
    final_email = negotiation_chain.run(
        financial_analysis=financial_analysis, 
        debtor_message=debtor_message
    )
    
    print("\n[FINAL SYSTEM OUTPUT - READY TO SEND]:")
    print(final_email)

# Run the test
if __name__ == "__main__":
    test_debtor = {"debt": 12500, "days_late": 180}
    test_message = "I just got laid off. I want to pay this medical bill but I have literally no money right now."
    
    process_debtor_account(test_debtor, test_message)
