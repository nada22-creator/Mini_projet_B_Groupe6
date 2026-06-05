# ============================================================================
# MODULE : integration_numerique.py
# DESCRIPTION : Implémentation des méthodes d'intégration numérique et calculs
#               analytiques pour l'évaluation de fonctions polynomiales.
# ============================================================================

# ============================================================================
# Importation de la bibLiothèque numpy pour l'utilisé dans la solution prochaine
# ============================================================================

import numpy as np


def f(x, p):
    """
    Évalue un polynôme du 3e degré pour une ou plusieurs valeurs de x.
    Modèle mathématique : f(x) = p1 + p2*x + p3*x^2 + p4*x^3

    Parameters:
    -----------
    x : float ou np.ndarray
        Le ou les points d'évaluation (abscisses).
    p : list ou np.ndarray
        Les coefficients du polynôme [p1, p2, p3, p4].

    Returns:
    --------
    float ou np.ndarray
        La valeur du polynôme f(x) (ordonnée).
    """
    return p[0] + p[1] * x + p[2] * (x ** 2) + p[3] * (x ** 3)


def solution_analytique(p, a, b):
    """
    Calcule la valeur exacte (théorique) de l'intégrale définie entre a et b
    en évaluant la primitive analytique F(x) : F(b) - F(a).
    Sert de référence absolue pour quantifier l'erreur des méthodes numériques.

    Parameters:
    -----------
    p : list
        Les coefficients du polynôme [p1, p2, p3, p4].
    a : float
        Borne inférieure de l'intégration.
    b : float
        Borne supérieure de l'intégration.

    Returns:
    --------
    float
        La valeur exacte de l'intégrale.
    """
    # Évaluation de la primitive F(x) à la borne supérieure b
    F_b = p[0] * b + (p[1] * (b ** 2)) / 2 + (p[2] * (b ** 3)) / 3 + (p[3] * (b ** 4)) / 4

    # Évaluation de la primitive F(x) à la borne inférieure a
    F_a = p[0] * a + (p[1] * (a ** 2)) / 2 + (p[2] * (a ** 3)) / 3 + (p[3] * (a ** 4)) / 4

    return F_b - F_a


# ==============================================================================
# FONCTION COMPLÉMENTAIRE : CALCUL DE L'ÉCART NUMÉRIQUE (ERREUR ABSOLUE)
# Cette fonction permet de quantifier précisément l'exactitude de nos approximations
# en mesurant la distance absolue entre le modèle théorique et le résultat calculé.
# ==============================================================================

def calcul_erreur(exact, numerique):
    """
    Détermine l'erreur absolue résiduelle entre le calcul théorique et l'approximation.
    Formule mathématique appliquée : | I_exact - I_numerique |

    Parameters:
    -----------
    exact : float
        La valeur exacte de l'intégrale obtenue par la solution analytique.
    numerique : float
        La valeur approximative obtenue par la méthode d'intégration numérique.

    Returns:
    --------
    float
        L'écart absolu (valeur toujours positive) servant à l'analyse de convergence.
    """
    # La fonction native abs() élimine le signe pour obtenir une grandeur d'erreur pure

    return abs(exact - numerique)


# ==============================================================================
# SECTION 2.1 : MÉTHODE DES RECTANGLES (POINT MILIEU)
# ==============================================================================

def rectangles_base(p, a, b, n):
    """
    Calcule l'intégrale numérique par la méthode des rectangles (point milieu)
    en utilisant uniquement des structures de boucles Python standard.

    Cette approche itérative évalue les sous-intervalles un par un, ce qui
    permet d'analyser le comportement algorithmique classique (boucle for).
    """
    # Calcul de la largeur constante (pas H) de chaque segment
    largeur = (b - a) / n
    somme_hauteurs = 0.0

    # Parcours itératif de chacun des n segments
    for i in range(n):
        # Positionnement de l'abscisse au centre géométrique du segment courant
        milieu = a + (i + 0.5) * largeur
        # Accumulation de la hauteur (valeur de la fonction au point milieu)
        somme_hauteurs += f(milieu, p)

    # L'aire totale est la somme des hauteurs multipliée par la largeur commune
    return somme_hauteurs * largeur


def rectangles_numpy(p, a, b, n):
    """
    Calcule l'intégrale numérique par la méthode des rectangles (point milieu)
    en utilisant la vectorisation de la bibliothèque NumPy.

    Aucune boucle 'for' ou 'while' n'est tolérée ici afin d'exploiter les calculs
    simultanés en mémoire .
    """
    # Calcul de la largeur constante de chaque segment

    largeur = (b - a) / n

    # Génération d'un tableau vectorisé contenant instantanément tous les milieux
    # On commence à (a + largeur/2) et on s'arrête à (b - largeur/2) avec exactement n points
    centres = np.linspace(a + largeur / 2, b - largeur / 2, n)

    # Évaluation simultanée de la fonction f(x) sur tout le tableau 'centres',
    # suivie de la somme de toutes les hauteurs et de la multiplication par la largeur
    return np.sum(f(centres, p)) * largeur