# Proyecto: Estudio Experimental de la Inercia Rotacional de una Masa Puntual

## Curso

Retos Científicos

## Integrantes del equipo

* David Felipe Chirivi Carreño
* Haniel Chaves Diaz
* Karen Tatiana Valero

---

## Selección del reto

El reto seleccionado consiste en la determinación experimental y teórica del **momento de inercia ($I$)** de una masa puntual. El proyecto utiliza un sistema de rotación con una plataforma, una polea de pasos y una masa colgante para aplicar un torque conocido.

A diferencia de los experimentos tradicionales, este enfoque integra **herramientas computacionales (Python)** para simular el comportamiento dinámico y contrastar los datos obtenidos mediante sensores o análisis de video con el modelo físico ideal.

---

## Objetivo general

Validar experimentalmente la expresión teórica $I = MR^2$ para una masa puntual, analizando la relación entre el torque aplicado y la aceleración angular resultante en un sistema rotacional.

---

## Alcance del proyecto

* **Montaje Físico:** Configuración del sistema rotacional (Base en "A", plataforma, polea y masas).
* **Modelado Matemático:** Aplicación de la Segunda Ley de Newton para traslación (masa colgante) y rotación (plataforma).
* **Simulación Computacional:** Desarrollo de un script en Python para predecir el movimiento del sistema bajo diferentes condiciones de masa y radio.
* **Análisis de Datos:** Medición de la aceleración angular ($\alpha$) y cálculo de la inercia experimental para su comparación con el valor teórico.

---
# Experimento 2 — Inercia Rotacional de una Masa Puntual
### PASCO ME-8950A · Complete Rotational System

> **Objetivo:** Determinar experimentalmente la inercia rotacional de una masa puntual y verificar que coincide con el valor teórico $I = MR^2$, aplicando un torque conocido y midiendo la aceleración angular resultante.

---

## 🎬 Referencias Audiovisuales

### Directamente relacionados con el experimento

| # | Título | Fuente | URL |
|---|--------|--------|-----|
| 1 | **Rotational Dynamics — Torque & Angular Acceleration** | PASCO Scientific | [pasco.com/resources/video/qYx7hag8bj4](https://www.pasco.com/resources/video/qYx7hag8bj4) |
| 2 | **Rotational Inertia** (disco, anillo y masa puntual con Rotary Motion Sensor) | PASCO Scientific | [pasco.com/resources/lab-experiments/458](https://www.pasco.com/resources/lab-experiments/458) |
| 3 | **Rotational Inertia of a Point Mass** | JoVE Science Education | [jove.com/es/v/10327](https://www.jove.com/es/v/10327/rotational-inertia-and-moment-of-intertia) |

### Apoyo conceptual

| # | Título | Fuente | URL |
|---|--------|--------|-----|
| 4 | **Más sobre momento de inercia** | Khan Academy (ES) | [es.khanacademy.org](https://es.khanacademy.org/science/physics/torque-angular-momentum/torque-tutorial/v/more-on-moment-of-inertia) |
| 5 | **Momento de inercia de objetos compuestos** | JoVE Science Education | [jove.com/es/science-education/v/12709](https://www.jove.com/es/science-education/v/12709/moment-of-inertia-of-compound-objects) |

---

## 📚 Bibliografía

### General — Libros de texto

**[1]** Serway, R. A., & Jewett, J. W. (2019). *Physics for Scientists and Engineers* (10th ed.). Cengage Learning.
> Capítulos 10 (*Rotation of a Rigid Object About a Fixed Axis*) y 11 (*Angular Momentum*).

**[2]** Halliday, D., Resnick, R., & Krane, K. (2014). *Physics* (5th ed., Vol. 1). John Wiley & Sons.
> Capítulo 12 (*Rotation of a Rigid Body About a Fixed Axis*).

**[3]** Young, H. D., & Freedman, R. A. (2020). *University Physics with Modern Physics* (15th ed.). Pearson Education.
> Capítulos 9 (*Rotation of Rigid Bodies*) y 10 (*Dynamics of Rotational Motion*).

**[4]** Tipler, P. A., & Mosca, G. (2008). *Physics for Scientists and Engineers* (6th ed.). W. H. Freeman and Company.
> Capítulo 9 (*Rotation*).

### Específica — Guías de laboratorio

**[5]** PASCO Scientific. (2009). *Instruction Manual and Experiment Guide: Complete Rotational System ME-8950A* (012-05293F). PASCO Scientific.
> Experimento 2: "Rotational Inertia of a Point Mass", pp. 15–18.
> 🔗 [cdn.pasco.com — ME-8950A Manual](https://cdn.pasco.com/product_document/Complete-Rotational-System-Manual-ME-8950A.pdf)

**[6]** Michigan State University — Department of Physics. (2015). *Lab 8: Rotational Motion — Moment of Inertia* (PHY251). Michigan State University.
> 🔗 [web.pa.msu.edu — Lab 8 PHY251](https://web.pa.msu.edu/courses/2015spring/PHY251/Lab_08.pdf)

**[7]** Mercer University — Department of Physics. (s.f.). *Moment of Inertia & Rotational Energy — Physics Lab IX*. Mercer University.
> 🔗 [physics.mercer.edu — Lab IX](https://physics.mercer.edu/labs/manuals/manualMECHlab/momInertiaRotEnergy/MofInertia_RotEnergy3.pdf)

# 📄 Documentos

| # | Documento | Extensión |
|---|-----------|-----------|
| 1 | **Retos** — Diseño y construcción de un riel de aire de bajo costo para el estudio experimental de colisiones unidimensionales | https://github.com/[usuario]/[repositorio]/blob/main/Retos_(1).pdf|
| 2 | **Estado del Arte y Posible Metodología Retos** — Dinámica rotacional: estado del arte y metodología propuesta para el Experimento 2 (PASCO ME-8950A) | https://github.com/user-attachments/files/25726622/ESTADO_DEL_ARTE_Y_POSIBLE_METODOLOGIA_RETOS.pdf |

# Montaje experimental
### Inercia Rotacional de una Masa Puntual · PASCO ME-8950A

## Partes del montaje

### Grupo 1 — Base y estructura

**1. Base tipo "A"**
- Hierro fundido (opción original PASCO)
- Acero estructural soldado con patas niveladoras *(recomendado por costo)*
- Aluminio grueso con patas de caucho

**2. Eje vertical rotatorio**
- Acero inoxidable torneado
- Acero al carbono con tratamiento superficial *(recomendado por costo)*
- Aluminio 6061 torneado

**3. Rodamientos de baja fricción**
- Rodamientos de bolas SKF o FAG (serie 619x)
- Rodamientos ABEC-5 o ABEC-7
- Cojinetes de bronce sinterizado *(recomendado por costo)*

**4. Anillos de retención tipo "E"**
- Anillos E de acero al carbono DIN 6799 *(recomendado por costo)*
- Anillos circlip de acero inoxidable

---

### Grupo 2 — Plataforma rotatoria

**5. Pista con ranura en T**
- Perfil de aluminio extruido 20x20 mm o 40x20 mm *(recomendado por costo)*
- Aluminio 6061-T6 mecanizado
- Acero inoxidable laminado

**6. Tornillo de ajuste y tuerca cuadrada**
- Tornillo moleteado M5 o M6 en acero inoxidable con tuerca cuadrada de acero *(recomendado por costo)*
- Tornillo de cabeza moleteada en nylon

**7. Varilla de montaje**
- Varilla roscada M8 de ferretería *(recomendado por costo)*
- Varilla de aluminio con rosca en un extremo
- Barra de soporte universal de laboratorio (rosca 1/4")

---

### Grupo 3 — Sistema de torque

**8. Polea redireccionadora**
- Polea de nylon impresa en 3D *(recomendado por costo)*
- Polea de plastico ABS con rodamiento interno
- Polea de aluminio mecanizado

**9. Hilo de suspension**
- Hilo de pesca monofilamento 8-10 lb *(recomendado por costo)*
- Hilo de nylon monofilamento (0.3-0.5 mm)
- Hilo de kevlar delgado

**10. Porta-masas y masas calibradas**
- Masas de acero inoxidable con gancho y porta-masas impreso en 3D *(recomendado por costo)*
- Set de masas de laton certificadas (1g, 2g, 5g, 10g, 20g, 50g)
- Masas de plomo

**11. Ajuste fino para masas menores a 1g**
- Clips metalicos de oficina *(recomendado por costo)*
- Arandelas pequenas M2 o M3
  
---

### Grupo 4 — Masa puntual

**14. Masa cuadrada (~300g)**
- Bloque de acero mecanizado en torneria local *(recomendado por costo)*
- Laton mecanizado
- Bloque de aluminio pesado
- Plomo (requiere precauciones de seguridad)

---

### Grupo 5 — Instrumentos auxiliares

**15. Balanza**
- Balanza digital de precision 0.1g *(recomendado por costo)*
- Balanza analitica de 0.01g de resolucion


## Consideraciones sobre el costo del montaje

Las tres decisiones que mayor impacto tienen sobre el costo total son reemplazar la interfaz PASCO por un Arduino, fabricar la polea mediante impresion 3D, y mandar tornear el eje y la masa puntual en una torneria local en lugar de adquirir las piezas originales de PASCO. Con estas tres elecciones, el montaje completo puede construirse por debajo de $150.000 COP.

Cabe mencionar que los cojinetes de bronce sinterizado, aunque introducen algo mas de friccion que los rodamientos de bolas, no representan un problema practico, ya que el procedimiento experimental contempla explicitamente la medicion y compensacion de la friccion residual mediante la determinacion de una masa de friccion.

---
## Estado del proyecto

* [x] Selección del reto
* [x] Diseño conceptual
* [ ] Diseño CAD
* [ ] Fabricación
* [ ] Experimentación
* [ ] Análisis de resultados
