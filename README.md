# AI Health Coach
<img src=""/>
# How to run it locally
run ollama locally with 

```bash
# run model
ollama pull llama3.2
ollama run llama3.2

# run backend
cd backend
uv sync
uv add main.py

# run frontend
bun install
bun --bun next dev
```
## Backend Architecture


## LLM Provider

I used llama3.2:3b parmaeters which was run by ollama , it was easy to inference locally , it is an slm but I have added context handling checks and methods to make it more comparable to real world use case.

## Tradeoffs 

- CTA(call to action): for severe issues it should prompt the user to consult a doctor which could be directed to a scheduling software internally 
- Vector db (Ideally the agent should search over a vector db with a query for context retrival per user as user's health concerns grow or more content is added like medical prescriptions (which could be parsed with ocr models), right now it's stored in db) , also pairing it up with semantic search and also tool calls for external knowledge on common remedies.
- Pragmatic nomeclature and code (my focus was majorly to make it functional , code should be more prgramatic further on , used llms to generate right now)
- Use cache for agent state , right now we are pulling agent state every time based on users query but it could be stored in cache for quick retrieval in production as users state updates also it could be flushed after a timeout , it will reloaded after a user starts a session basically reducing db calls for read but writes would have to be done.
- Fact checking and content filtering : the agent should reject symptoms that seem to be wrong or not necessarily valid for the user , like someone can say they have aids they should pick that up during a conversation or do quizzes in between to verify that a particular issue has been cured flag it as not active.
- notifications: if user reported a severe issue or the agent suggested a health checkup , it should add to queue to do a quick health quiz with maybe AWS SNS for the symptoms they shared in particular , because the agent state could become corrupted due to that