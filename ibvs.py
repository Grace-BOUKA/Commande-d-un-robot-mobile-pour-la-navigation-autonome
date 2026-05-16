import numpy as np

class Controlleur_IBVS:
    def __init__(self, camera, gain):
        self.cam = camera
        self.lmbda = gain 
        self.s_star = None
        self.Z_star = 0.5# Profondeur Z initialisée
        
        # Mémoire pour le lissage des entrées de commandes
        self.v_prev = 0.0
        self.w_prev = 0.0
        self.w_cam_prev = 0.0


    # Tri des points cibles s* dans le bon ordre dans la camera
    def Trier_points(self, points):
        pts = np.array(points)
        pts = pts[pts[:, 1].argsort()]
        Haut = pts[:2][pts[:2][:, 0].argsort()]
        Bas = pts[2:][pts[2:][:, 0].argsort()]
        return np.array([Haut[0], Haut[1], Bas[1], Bas[0]]).flatten()  
    
    
    # Commandes : v_robot, w_robot, w_camera
    def commande_IBVS(self, T_ech, visible_targets, robot_pose=None, obstacles_list=None):
        
        # Rotation de la camera si cible non visible
        if not visible_targets:
            return 0.0, 0.0, 0.5, None, None ,None

        # Extraction des pixels de la cible s en temps réels
        pixels_bruts = visible_targets[0][1]
        if len(pixels_bruts) != 4: return 0.0, 0.0, 0.5, None, None ,None # Cas cible non visible
        
        
        # ======================================================================
        #           Commande IBVS normale (sans evitement d'obstacle) 
        # ======================================================================
        
        # Aquisition des pixels de la cible courante s
        s = self.Trier_points(pixels_bruts) # Cible s
        
        # Calcul de l'erreur visuelle e = (s - s*)
        error = s - self.s_star
        x_err = np.linalg.norm(error) # Norme de l'erreur pour la zone morte
        
        L_s = []
        
        # Matrice d'interaction 8x3
        u0, v0, f = self.cam.u0, self.cam.v0, self.cam.focal
        d = self.cam.a     # distance centre_robot -> centre_R_platine
        alpha = self.cam.pan_angle # Rotation de la platine

        # Jacobienne de la platine : (Expressions dans Rc)
        J_platine = np.array([
            [np.sin(alpha), -d * np.cos(alpha), 0], # Vitesse  Vx (R_c)
            [ np.cos(alpha),  d * np.sin(alpha), 0], # Vitesse Vz (R_c)
            [ 0,              1 ,                 1]  # Rotation Wy (R_c)
        ])
        
        Z = self.Z_star # Profondeur de la cible estimée
        
        for i in range(0, 8, 2):
            u, v = s[i] - u0, s[i+1] - v0
            
            # Matrice d'interaction image Tc aux 3 DDL du robot
            T_c = np.array([
                [-f / Z,  u / Z, (u**2 + f**2) / f],
                [ 0,      v / Z, (u * v) / f]
            ])
            
            # Matrice d'interaction globale L_eq = (T_c * J_platine)
            L_eq = T_c @ J_platine
            L_s.append(L_eq[0])
            L_s.append(L_eq[1])
        
        L_s = np.array(L_s)
                   
        # Loi de commande IBVS (avec gain proportionnel)
        q_point = -self.lmbda * np.linalg.pinv(L_s) @ error
        
        v_c_ibvs = q_point[0]
        w_c_ibvs = q_point[1]
        w_cam_ibvs = q_point[2]
        
        # =====================================================================================
        
        

        # =====================================================================================
        #         Evitement d'obstacles (Champ de potentiel rotatif & Découplage)
        # =====================================================================================
        
        # Commandes pour la repulsion
        v_rep = 0.0 
        w_rep = 0.0
        
        # Initialisation pour le traitement de l'évitement d'obstacle
        v_c_total = v_c_ibvs 
        w_c_total = w_c_ibvs
        
        if robot_pose is not None and obstacles_list is not None: # Si presence d'obstacles
            x_r, y_r, theta_r = robot_pose
            d_influence = 1  # Distance d'anticipation
            k_w_evit = 1.5   # Gain de rotation pour s'écarter de l'obstacle
            k_v_frein = 0.5   # Gain de freinage
            largeur_robot = 0.2 # Estimation de la largeur du robot pour la planification

            for ox, oy, r_obs in obstacles_list:
                dx = ox - x_r
                dy = oy - y_r
                dist = np.sqrt(dx**2 + dy**2) # Distance centre obstacle - centre du robot
                dist_bord = dist - r_obs - largeur_robot # Distance bord obstacle - bord du robot
                

                if 0 < dist_bord < d_influence:
                    angle_obs_global = np.arctan2(dy, dx)
                    angle_obs_local = (angle_obs_global - theta_r + np.pi) % (2 * np.pi) - np.pi # Difference angle obstacle-robot em modulo 2pi
                    
                   
                    # ----------------------- Champs de potentiel repulsif ----------------------------
                    
                    if abs(angle_obs_local) < (np.pi / 1.5): # Si l'obstacle est devant
                        # On force un côté si c'est pile en face pour éviter les vibrations
                        glisser = -np.sign(angle_obs_local) if abs(angle_obs_local) > 0.1 else 1.0
                            
                        # Intensité de la répulsion
                        force = ((1.0 / dist_bord) - (1.0 / d_influence))
                        
                        # Freinage linéaire
                        v_rep -= np.cos(angle_obs_local) * force * k_v_frein
                        
                        # Rotation d'évitement (pour contourner)
                        w_rep += glisser * force * k_w_evit
                        
                        # Calcul des vitesses finales combinées (IBVS + Évitement) 
                        sigma= np.clip((d_influence - dist_bord) / d_influence,0,1)**2
                        v_c_total = (1-sigma)*v_c_ibvs + sigma*v_rep
                        w_c_total = (1-sigma)*w_c_ibvs + sigma*w_rep
                        
                        # Correction du point mort (si le robot n'avance plus)
                        if v_c_total < 0.05:
                            w_c_total = glisser * k_w_evit * 1.2
                        
                        break # On gère l'obstacle le plus proche en priorité et on sort
                    

        # Decouplage platine-robot : La platine compense la rotation du robot pour fixer la cible
        w_cam_total = w_cam_ibvs - (w_c_total - w_c_ibvs)
        # =============================================================================================
        
        
        #  Calcul des entrées de commandes avec saturation et lissage
        max_dv = 1.0 * T_ech  # Ecart de vitesse (m/s)
        max_dw = 1.5 * T_ech  # Ecart de vitesse de rotation (rad/s)
        v_c = np.clip(self.v_prev + np.clip(v_c_total - self.v_prev, -max_dv, max_dv), -0.8, 0.8)
        w_c = np.clip(self.w_prev + np.clip(w_c_total - self.w_prev, -max_dw, max_dw), -1.0, 1.0)
        w_cam = np.clip(self.w_cam_prev + np.clip(w_cam_total - self.w_cam_prev, -max_dw, max_dw), -1.5, 1.5)
        
        
        # Gestion de la zone morte
        if x_err < 10: # pixels            
            v_c *= 0.5
            w_c *= 0.5
            w_cam *= 0.5
        
            
        # Mise à jour de la mémoire pour la prochaine itération (Lissage)
        self.v_prev, self.w_prev, self.w_cam_prev = v_c, w_c, w_cam
        
        
        # Condition d'arret du robot
        if x_err < 3.0: # Stop
            v_c, w_c, w_cam, x_err , error = 0, 0, 0, 0, np.array([0,0,0,0,0,0,0,0])
            
        return v_c, w_c, w_cam, x_err, error,s