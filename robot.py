import numpy as np

class Robot_Differentiel:
    def __init__(self, x=0, y=0, theta=0):
        # Parametres géometriques du robot différentiel
        self.x = x
        self.y = y
        self.theta = theta
        self.R_roue = 0.3 # Rayon des roues (R)
        self.entraxe = 0.4 # Entraxe
    
    # Deplacement du robot
    def move(self, v_g, v_d, T_ech):
        # Modele cinematique du robot différentiel
        v_lineaire = (self.R_roue / 2) * (v_d + v_g)
        v_angulaire = (self.R_roue / self.entraxe) * (v_d - v_g)
        
        # Modele géometrique du robot différentiel
        self.x += v_lineaire * np.cos(self.theta) * T_ech
        self.y += v_lineaire * np.sin(self.theta) * T_ech
        self.theta += v_angulaire * T_ech
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi # Modulo 2pi
    
    # Coordonnées primitives du robot
    def get_pose(self):
        return np.array([self.x, self.y, self.theta])