from evaluation import evaluate_agent_scalar
from solucion import Agent

# seeds de prueba (si tu profe usa seeds fijas, aquí pondrás esas)
seeds = list(range(50))

agent = Agent()
res = evaluate_agent_scalar(agent, seeds, size=4, max_steps=5000)
print(res)