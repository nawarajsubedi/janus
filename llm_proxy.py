import asyncio

from fastapi import Body, FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return "Ok"


@app.post("/llm-proxy")
async def llm_proxy(question: str = Body(..., embed=True)):

    await asyncio.sleep(1)
    return {
        "Status": "Request coming in LLM Proxy server",
        "question": question,
        # "answer": "It is a response from LLM-Proxy",
    }
