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

Asumiendo que la solución está en `submission.py`:

```bash
python eval_student.py --agent-module submission --episodes 5
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
├── submission.py          # Implementación del agente
├── run_2048.py          # Runner (manual y agente)
├── evaluation.py        # Evaluación oficial
├── eval_student.py        # Evaluación local
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
