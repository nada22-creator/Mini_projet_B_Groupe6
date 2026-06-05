# ==============================================================================
# SOLUTIONS DU GROUPE 6 : CODE PRINCIPAL DE CONTRÔLE ET D'ANALYSE AVANCÉE
# MODULE PRINCIPAL : main.py
# DESCRIPTION : Analyse de performance (timeit), étude de convergence 2D log-log
#               et cartographie 3D de la sensibilité de l'erreur absolue.
# ==============================================================================

import timeit
import numpy as np
import matplotlib.pyplot as plt
import integration_numerique as int_num

# ==============================================================================
# CONFIGURATION DES PARAMÈTRES DU PROBLÈME (POLYNÔME ET BORNES)
# ==============================================================================
# Coefficients [p1, p2, p3, p4] du polynôme du 3e degré à intégrer
coefficients = [1.0, 2.0, 3.0, 0.5]
borne_a = -2.0
borne_b_fixe = 3.0

# ==============================================================================
# PHASE 1 : VALIDATION MATHÉMATIQUE INITIALE (Pour n = 10 segments)
# ==============================================================================
n_validation = 10

# Calcul des approximations et de la solution exacte de référence
valeur_exacte_fixe = int_num.solution_analytique(coefficients, borne_a, borne_b_fixe)
res_base = int_num.rectangles_base(coefficients, borne_a, borne_b_fixe, n_validation)
res_numpy = int_num.rectangles_numpy(coefficients, borne_a, borne_b_fixe, n_validation)

print("=" * 80)
print("             VALEURS OBTENUES POUR LA VALIDATION INITIALE (n = 10)      ")
print("=" * 80)
print(f"Solution Analytique (Théorique) : {valeur_exacte_fixe:.6f}")
print(
    f"Méthode des Rectangles (Base)   : {res_base:.6f} (Erreur: {int_num.calcul_erreur(valeur_exacte_fixe, res_base):.4e})")
print(
    f"Méthode des Rectangles (NumPy)  : {res_numpy:.6f} (Erreur: {int_num.calcul_erreur(valeur_exacte_fixe, res_numpy):.4e})")
print("-" * 80 + "\n")

# ==============================================================================
# PHASE 2 : MESURE DES PERFORMANCES DU CODE AVEC TIMEIT
# ==============================================================================
# Utilisation d'un n plus élevé pour solliciter le processeur et valider l'optimisation
n_benchmark = 2000
nb_repetitions = 100

print("=" * 80)
print(f"       CHRONOMÉTRAGE DES PERFORMANCES AVEC TIMEIT (n = {n_benchmark})")
print("=" * 80)

# Mesure du temps pour la structure itérative pure (Python de base)
temps_base = timeit.timeit(
    lambda: int_num.rectangles_base(coefficients, borne_a, borne_b_fixe, n_benchmark),
    number=nb_repetitions
)

# Mesure du temps pour l'approche vectorisée (NumPy)
temps_numpy = timeit.timeit(
    lambda: int_num.rectangles_numpy(coefficients, borne_a, borne_b_fixe, n_benchmark),
    number=nb_repetitions
)

print(f"Temps cumulé - Boucle Python standard : {temps_base:.5f} secondes")
print(f"Temps cumulé - Vectorisation NumPy     : {temps_numpy:.5f} secondes")
print(f"Rapport d'efficacité : NumPy est {temps_base / temps_numpy:.1f}x plus rapide.")
print("-" * 80 + "\n")

# ==============================================================================
# PHASE 3 : COLLECTE DES DONNÉES ET CONFIGURATION DES GRAPHIQUES (2D ET 3D)
# ==============================================================================
print("=" * 80)
print("             GÉNÉRATION DU TABLEAU DE BORD DES GRAPHIQUES (2D & 3D)     ")
print("=" * 80)

# --- Préparation des données pour le graphique 2D ---
liste_n = [10, 50, 100, 500, 1000, 5000]
erreurs_2d = []

for n in liste_n:
    approx_numpy = int_num.rectangles_numpy(coefficients, borne_a, borne_b_fixe, n)
    erreurs_2d.append(int_num.calcul_erreur(valeur_exacte_fixe, approx_numpy))

# --- Préparation des données pour la surface 3D (Analyse de sensibilité) ---
# On fait varier simultanément la borne b et le nombre de segments n
bornes_b_dynamiques = np.linspace(0.5, 4.0, 30)
segments_n_dynamiques = np.array([10, 20, 50, 100, 200, 500])

# Génération d'une grille de coordonnées 2D à partir des deux vecteurs de paramètres
B, N = np.meshgrid(bornes_b_dynamiques, segments_n_dynamiques)
Z_erreur = np.zeros_like(B)

# Calcul de l'erreur absolue pour chaque nœud de la grille (b, n)
for i in range(len(segments_n_dynamiques)):
    for j in range(len(bornes_b_dynamiques)):
        n_courant = int(N[i, j])
        b_courant = B[i, j]

        exact = int_num.solution_analytique(coefficients, borne_a, b_courant)
        approx = int_num.rectangles_numpy(coefficients, borne_a, b_courant, n_courant)
        Z_erreur[i, j] = int_num.calcul_erreur(exact, approx)

# Initialisation de la figure principale contenant les deux sous-graphiques côte à côte
fig = plt.figure(figsize=(16, 6.5))

# ------------------------------------------------------------------------------
# SOUS-GRAPHIQUE 1 : Courbe de Convergence 2D (Log-Log)
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(liste_n, erreurs_2d, 's--', label="Méthode des Rectangles (NumPy)", color="royalblue", linewidth=1.5)

# Passage aux échelles logarithmiques pour observer l'ordre de convergence
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_title("Évolution de l'erreur absolue (Échelle Log-Log)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Nombre de segments de discrétisation (n)", fontsize=10)
ax1.set_ylabel("Erreur absolue commise |I_exact - I_num|", fontsize=10)
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend(loc="best")

# ------------------------------------------------------------------------------
# SOUS-GRAPHIQUE 2 : Cartographie et Surface d'Erreur 3D
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Tracé de la surface d'erreur en appliquant une transformation Log10 pour l'échelle
# Utilisation de la palette thermique 'viridis' pour illustrer l'intensité
surface = ax2.plot_surface(B, np.log10(N), np.log10(Z_erreur), cmap='viridis', edgecolor='none', alpha=0.9)

# Configuration et étiquetage des axes tridimensionnels
ax2.set_title("Cartographie 3D de la sensibilité de l'erreur", fontsize=11, fontweight='bold')
ax2.set_xlabel("Borne supérieure du domaine (b)", fontsize=9)
ax2.set_ylabel("Densité du maillage : Log10(n)", fontsize=9)
ax2.set_zlabel("Amplitude de l'erreur : Log10(Erreur)", fontsize=9)

# Ajout d'une barre colorée (Colorbar) explicative pour les niveaux d'intensité
fig.colorbar(surface, ax=ax2, shrink=0.5, aspect=12, label="Niveau d'erreur (Échelle Log10)")

# Ajustement de l'angle de vue tridimensionnel (Élévation, Azimut) pour maximiser le rendu
ax2.view_init(elev=22, azim=-125)

# ------------------------------------------------------------------------------
# ENREGISTREMENT ET AFFICHAGE DU TABLEAU DE BORD FINALE
# ------------------------------------------------------------------------------
plt.tight_layout()
nom_rendu_image = "analyse_integration_2D_3D.png"
plt.savefig(nom_rendu_image, dpi=300, bbox_inches='tight')

print(f"Succès ! Tableau de bord visuel sauvegardé sous : '{nom_rendu_image}'")
print("Ouverture de l'interface d'affichage en cours...")
print("=" * 80)

plt.show()