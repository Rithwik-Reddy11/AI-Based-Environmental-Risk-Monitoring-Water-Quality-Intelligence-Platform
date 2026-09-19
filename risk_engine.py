import numpy as np


def calculate_wqi(pH, do, nitrate, turbidity, temp):
    pH_score = max(0.0, 100 - abs(pH - 7) * 15)
    do_score = np.clip((do - 4) * 25, 0, 100)
    nitrate_score = max(0.0, 100 - nitrate * 5)
    turbidity_score = max(0.0, 100 - turbidity * 4)
    temp_penalty = min(abs(temp - 22) * 1.5, 30)
    score = do_score * .35 + pH_score * .25 + nitrate_score * .20 + turbidity_score * .15 + (100 - temp_penalty) * .05
    return float(np.clip(score, 0, 100))


def risk_from_probability(probability):
    p = float(np.clip(probability, 0, 100))
    if p <= 20: return "Very Low Risk", "LOW"
    if p <= 40: return "Low Risk", "LOW"
    if p <= 60: return "Moderate Risk", "MODERATE"
    if p <= 80: return "High Risk", "HIGH"
    return "Very High Risk", "CRITICAL"


def risk_score(probability, wqi):
    return round(float(np.clip(0.7 * probability + 0.3 * (100 - wqi), 0, 100)), 1)


def what_if_message(base, changed):
    delta = changed - base
    if delta >= 10: return f"Risk increased by {delta:.1f} points — investigate the changed conditions."
    if delta <= -10: return f"Risk decreased by {abs(delta):.1f} points — conditions are improving."
    return "The scenario produces a relatively small change in estimated risk."
