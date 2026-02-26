# 🧠 Laboratorio 3 – Agente 2048

Implementación de un agente para el juego **2048**.

El objetivo del laboratorio es diseñar una política que maximice el desempeño promedio del agente bajo seeds fijas.

---

## 📌 Requerimientos

- Python 3.10+
- numpy < 2
- matplotlib


## ▶️ Cómo ejecutar

### 🎮 Modo manual
Permite jugar 2048 con teclado.

```bash
python run_2048.py --mode manual
```

---

### 🤖 Modo agente

Asumiendo que la solución está en `solucion.py`:

```bash
python run_2048.py --mode agent --agent-module solucion --agent-class Agent --episodes 50
```

---

## 📊 Evaluación oficial

El laboratorio usa `evaluation.py`:

```python
final_score = 1000 * mean_log_score \
            + 30 * mean_log2_max_tile \
            + 10 * median_log_score \
            - 2 * mean_log_steps
```

Donde:

- `mean_log_score = mean(log(1 + score))`
- `mean_log2_max_tile = mean(log2(max_tile))`
- `mean_log_steps = mean(log(1 + steps))`

Para evaluación local:

```python
from evaluation import evaluate_agent_scalar
from solucion import Agent

seeds = [0] #list(range(50))
agent = Agent()
print(evaluate_agent_scalar(agent, seeds))
```

---

## 🚫 Uso de VRAM

El agente está implementado únicamente con **NumPy (CPU)**.

No utiliza GPU ni frameworks como PyTorch o TensorFlow.

Consumo de VRAM: **~0 MB**  
Cumple con la restricción de máximo 5GB en inferencia.

---

## 📂 Estructura del proyecto

```
.
├── solucion.py          # Implementación del agente
├── run_2048.py          # Runner (manual y agente)
├── evaluation.py        # Evaluación oficial
├── eval_local.py        # Evaluación local
├── game_2048.py         # Lógica del juego
├── viz_2048.py          # Renderizado
```

---

## 🎯 Objetivo

Maximizar `final_score` en múltiples seeds fijas bajo restricciones de eficiencia.

---

## 👤 Grupo - OptimusPrime:

* César Eduardo Pajuelo Reyes
* Gonzalo Alonso Rodriguez Gutierrez
