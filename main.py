# ==============================================================================
# SOLUTIONS DU GROUPE 6 : CODE PRINCIPAL DE CONTRÔLE ET D'ANALYSE
# MINI-PROJET B - EVALUATION DES PERFORMANCES ET VISUALISATION DE LA CONVERGENCE
# ==============================================================================

import timeit
import matplotlib.pyplot as plt
import integration_numerique as int_num

# ==============================================================================
# CONFIGURATION DES PARAMÈTRES DU PROBLÈME (ÉQUATION ET BORNES)
# ==============================================================================
# Coefficients [p1, p2, p3, p4] du polynôme du 3e degré à intégrer
coefficients = [1.0, 2.0, 3.0, 0.5]
borne_a = -2.0
borne_b = 3.0

# ==============================================================================
# PHASE 1 : VALIDATION MATHÉMATIQUE INITIALE (Pour n = 10 segments)
# ==============================================================================
n_validation = 10

# Calcul des approximations et de la solution exacte de référence
valeur_exacte = int_num.solution_analytique(coefficients, borne_a, borne_b)
res_base = int_num.rectangles_base(coefficients, borne_a, borne_b, n_validation)
res_numpy = int_num.rectangles_numpy(coefficients, borne_a, borne_b, n_validation)

print("=" * 70)
print("             VALEURS OBTENUES POUR LA VALIDATION INITIALE (n = 10)      ")
print("=" * 70)
print(f"Solution Analytique (Théorique) : {valeur_exacte:.6f}")
print(
    f"Méthode des Rectangles (Base)   : {res_base:.6f} (Erreur: {int_num.calcul_erreur(valeur_exacte, res_base):.4e})")
print(
    f"Méthode des Rectangles (NumPy)  : {res_numpy:.6f} (Erreur: {int_num.calcul_erreur(valeur_exacte, res_numpy):.4e})")
print("-" * 70 + "\n")

# ==============================================================================
# PHASE 2 : MESURE DES PERFORMANCES DU CODE AVEC TIMEIT
# ==============================================================================
# Utilisation d'un n plus élevé pour solliciter le processeur et voir l'impact
n_benchmark = 2000
nb_repetitions = 100

print("=" * 70)
print(f"       CHRONOMÉTRAGE DES PERFORMANCES AVEC TIMEIT (n = {n_benchmark})")
print("=" * 70)

# Mesure du temps pour la structure itérative pure (Python de base)
temps_base = timeit.timeit(
    lambda: int_num.rectangles_base(coefficients, borne_a, borne_b, n_benchmark),
    number=nb_repetitions
)

# Mesure du temps pour l'approche vectorisée (NumPy)
temps_numpy = timeit.timeit(
    lambda: int_num.rectangles_numpy(coefficients, borne_a, borne_b, n_benchmark),
    number=nb_repetitions
)

print(f"Temps cumulé - Boucle Python standard : {temps_base:.5f} secondes")
print(f"Temps cumulé - Vectorisation NumPy     : {temps_numpy:.5f} secondes")
print(f"Rapport d'efficacité : NumPy est {temps_base / temps_numpy:.1f}x plus rapide.")
print("-" * 70 + "\n")

# ==============================================================================
# PHASE 3 : COLLECTE DES DONNÉES ET TRAÇAGE MATPLOTLIB (CONVERGENCE LÉGALE)
# ==============================================================================
print("=" * 70)
print("             GÉNÉRATION DU GRAPHIQUE DE CONVERGENCE DES ERREURS         ")
print("=" * 70)

# Liste des subdivisions (n) demandées pour l'analyse
liste_n = [10, 50, 100, 500, 1000, 5000]
erreurs_base = []
erreurs_numpy = []

# Évaluation des erreurs pour chaque niveau de raffinement de maillage
for n in liste_n:
    # Récupération de l'erreur absolue pour la méthode itérative
    approx_base = int_num.rectangles_base(coefficients, borne_a, borne_b, n)
    erreurs_base.append(int_num.calcul_erreur(valeur_exacte, approx_base))

    # Récupération de l'erreur absolue pour la méthode vectorisée
    approx_numpy = int_num.rectangles_numpy(coefficients, borne_a, borne_b, n)
    erreurs_numpy.append(int_num.calcul_erreur(valeur_exacte, approx_numpy))

# Initialisation et configuration de la figure Matplotlib
plt.figure(figsize=(9, 5.5))

# Tracé des points et lignes d'erreurs
plt.plot(liste_n, erreurs_base, 'o-', label="Python Standard (Boucle)", color="crimson", linewidth=1.5)
plt.plot(liste_n, erreurs_numpy, 's--', label="NumPy Vectorisé", color="royalblue", linewidth=1.5)

# Passage en échelle logarithmique log-log (Fondamental en analyse numérique)
plt.xscale('log')
plt.yscale('log')

# Définition des labels et du titre du graphique
plt.title("Analyse comparative de la convergence : Méthode des Rectangles", fontsize=12, fontweight='bold')
plt.xlabel("Nombre de segments de discrétisation (n)", fontsize=10)
plt.ylabel("Erreur absolue commise |I_exact - I_num|", fontsize=10)

# Configuration de la grille (principale et secondaire pour l'échelle log)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(loc="best", frameon=True)

# Enregistrement de la figure en haute résolution pour le livrable (rapport écrit)
nom_fichier_image = "convergence_erreur_rectangles.png"
plt.savefig(nom_fichier_image, dpi=300, bbox_inches='tight')

print(f"Succès ! Graphique sauvegardé localement sous : '{nom_fichier_image}'")
print("Affichage de la fenêtre graphique en cours...")
print("=" * 70)

plt.show()