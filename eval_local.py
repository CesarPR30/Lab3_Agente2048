from evaluation import evaluate_agent_scalar
from solucion import Agent

seeds = [0] #list(range(50))

agent = Agent()
res = evaluate_agent_scalar(agent, seeds, size=4, max_steps=5000)
print(res)