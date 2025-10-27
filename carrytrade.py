import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = "e16626c91fa2b1af27704a783939bf72"  # À remplacer
fred = Fred(api_key=API_KEY)

# Dates d'analyse
START_DATE = '2015-01-01'
END_DATE = '2023-12-31'

# =============================================================================
# 1. TÉLÉCHARGER LES DONNÉES
# =============================================================================

print("Téléchargement des données depuis FRED...")

# Taux d'intérêt USD (domestique) - Treasury 3 mois
i_t = fred.get_series('DGS3MO', observation_start=START_DATE, observation_end=END_DATE)

# Taux d'intérêt EUR (étranger) - Taux 3 mois
i_t_star = fred.get_series('IR3TIB01EUM156N', observation_start=START_DATE, observation_end=END_DATE)

# Taux de change USD/EUR
S_t = fred.get_series('DEXUSEU', observation_start=START_DATE, observation_end=END_DATE)

# =============================================================================
# 2. PRÉPARER LES DONNÉES
# =============================================================================

print("Préparation des données...")

# Combiner dans un DataFrame
df = pd.DataFrame({
    'i_usd': i_t,
    'i_eur': i_t_star,
    'exchange_rate': S_t
})

# Nettoyer (supprimer les valeurs manquantes)
df = df.dropna()

# Convertir les taux en décimaux (de % à décimal)
df['i_usd'] = df['i_usd'] / 100
df['i_eur'] = df['i_eur'] / 100

# Rééchantillonner mensuellement (prendre la dernière valeur du mois)
df = df.resample('M').last().dropna()

print(f"\nPremières lignes des données:")
print(df.head())

# =============================================================================
# 3. CALCULER LE PAYOFF ET LE CARRY TRADING
# =============================================================================

print("\nCalcul des rendements...")

# Rendement du taux de change: (S_{t+1} / S_t) - 1
df['fx_return'] = df['exchange_rate'].pct_change()

# Payoff de la position LONG sur EUR:
# z*_{t+1} = (1 + i*_t) * (S_{t+1}/S_t) - (1 + i_t)
df['z_star'] = (1 + df['i_eur']) * (1 + df['fx_return']) - (1 + df['i_usd'])

# Signal de Carry Trading: sign(i*_t - i_t)
df['interest_diff'] = df['i_eur'] - df['i_usd']
df['signal'] = np.sign(df['interest_diff'])

# Rendement du Carry Trading:
# z^C_{t+1} = sign(i*_t - i_t) * z*_{t+1}
# On utilise le signal du mois précédent (shift(1))
df['carry_return'] = df['signal'].shift(1) * df['z_star']

# Supprimer la première ligne (pas de rendement)
df = df.dropna()

# =============================================================================
# 4. CALCULER LES PERFORMANCES
# =============================================================================

print("\nCalcul des performances...")

# Rendements cumulés (valeur du portefeuille)
df['cumul_carry'] = (1 + df['carry_return']).cumprod()
df['cumul_long'] = (1 + df['z_star']).cumprod()

# Métriques de performance
total_return_carry = df['cumul_carry'].iloc[-1] - 1
total_return_long = df['cumul_long'].iloc[-1] - 1

annual_return_carry = df['carry_return'].mean() * 12
annual_vol_carry = df['carry_return'].std() * np.sqrt(12)
sharpe_carry = annual_return_carry / annual_vol_carry

print("\n" + "="*60)
print("RÉSULTATS")
print("="*60)
print(f"\nStratégie CARRY TRADING:")
print(f"  Rendement total: {total_return_carry*100:.2f}%")
print(f"  Rendement annualisé: {annual_return_carry*100:.2f}%")
print(f"  Volatilité annualisée: {annual_vol_carry*100:.2f}%")
print(f"  Ratio de Sharpe: {sharpe_carry:.2f}")

print(f"\nStratégie LONG EUR (sans signal):")
print(f"  Rendement total: {total_return_long*100:.2f}%")

print(f"\nDifférence: {(total_return_carry - total_return_long)*100:.2f}%")

# =============================================================================
# 5. VISUALISATIONS
# =============================================================================

print("\nCréation des graphiques...")

# Figure avec 3 graphiques
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Graphique 1: Différentiel de taux d'intérêt
axes[0].plot(df.index, df['interest_diff'] * 100, linewidth=2, color='blue')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('Différentiel de Taux d\'Intérêt (EUR - USD)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Différentiel (%)')
axes[0].grid(True, alpha=0.3)

# Graphique 2: Signal de trading
colors = ['red' if x < 0 else 'green' for x in df['signal']]
axes[1].bar(df.index, df['signal'], color=colors, alpha=0.6)
axes[1].set_title('Signal de Trading (Long EUR si > 0, Short EUR si < 0)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Signal')
axes[1].grid(True, alpha=0.3)

# Graphique 3: Rendements cumulés
axes[2].plot(df.index, df['cumul_carry'], label='Carry Trading', linewidth=2, color='green')
axes[2].plot(df.index, df['cumul_long'], label='Long EUR (sans signal)', linewidth=2, color='orange', alpha=0.7)
axes[2].set_title('Évolution du Portefeuille (1 USD initial)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Valeur du Portefeuille')
axes[2].set_xlabel('Date')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('carry_trading_results.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegardé: carry_trading_results.png")
plt.show()

# =============================================================================
# 6. TABLEAU RÉCAPITULATIF
# =============================================================================

print("\n" + "="*60)
print("INTERPRÉTATION POUR VOS ÉLÈVES")
print("="*60)
print("""
1. DIFFÉRENTIEL DE TAUX:
   - Quand i_EUR > i_USD → Signal LONG sur EUR (on achète EUR)
   - Quand i_EUR < i_USD → Signal SHORT sur EUR (on vend EUR)

2. PAYOFF (z*):
   - C'est le gain/perte d'investir en EUR vs rester en USD
   - Formule: (1 + i_EUR) × (S_{t+1}/S_t) - (1 + i_USD)

3. CARRY TRADING (z^C):
   - On suit le signal basé sur le différentiel de taux
   - Formule: sign(i_EUR - i_USD) × z*

4. RÉSULTAT:
   - Le carry trading exploite les différences de taux d'intérêt
   - Il fonctionne si le taux de change ne compense pas totalement
     le différentiel de taux (violation de la parité des taux)
""")

print("\nFIN DE L'ANALYSE")