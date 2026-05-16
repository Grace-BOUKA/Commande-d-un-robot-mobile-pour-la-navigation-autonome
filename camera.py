import numpy as np
import cv2

class Camera_Virtuel:
    def __init__(self, focal=100, resolution=(320, 240)):
     # Parametres intrinseques de la camera
        self.resolution = resolution
        self.width, self.height = resolution
        self.fov = np.radians(focal)
        self.focal = (self.width / 2) / np.tan(self.fov / 2)
        #Centres optiques
        self.u0 = self.width / 2
        self.v0 = self.height / 2
        
        self.pan_angle = 0.0 # Angle de rotation propre de la platine (la caméra hérite la rotation de la platine)
        self.a = 0.3  # Décalage centre robot <--> centre platine
        self.h_cam = 0.05 # Hauteur de la camera par rapport à la base de la platine
        
        
    
    # --------------- Passage coordonnées primitves en coordonnées pixel
    def changement_repere(self, point_3d, robot_pose):
        cible_x, cible_y, cible_z = point_3d
        cible = np.array([cible_x, cible_y, cible_z, 1]).reshape(4, 1)
        robot_x, robot_y, robot_theta = robot_pose

        # R0 -> R_robot
        T_OR = np.array([
            [np.cos(robot_theta), -np.sin(robot_theta), 0, robot_x],
            [np.sin(robot_theta),  np.cos(robot_theta), 0, robot_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # R_robot -> R_platine
        T_Rpl = np.array([
            [np.cos(self.pan_angle), -np.sin(self.pan_angle), 0, self.a],
            [np.sin(self.pan_angle),  np.cos(self.pan_angle), 0, 0],
            [0, 0, 1, -self.h_cam],
            [0, 0, 0, 1]
        ])
        
        # R_platine -> R_caméra (avec Ajustement des axes selon la vue OpenCV)
        T_plc = np.array([
            [ 0,  0,  1, 0], # X_platine = Z_cam (La visée optique )
            [-1,  0,  0, 0], # Y_platine = -X_cam 
            [ 0, -1,  0, 0], # Z_platine = -Y_cam
            [ 0,  0,  0, 1]
        ])

        # Matrice de passage complète Camera -> Monde
        T_OC = T_OR @ T_Rpl @ T_plc
        
        # Transformation : on passe du repère Monde au repère Camera (Inverse)
        pose_cible = np.linalg.inv(T_OC) @ cible
        
        xc, yc, zc, _ = pose_cible.flatten()

        # Gestion angle mort
        if zc <= 0.1: # la cible est derrière ou trop près
            return None 
        
        # Projection en perspective (Modèle de la caméra pinhole): Coordonnées métriques
        u = self.focal * (xc / zc) + self.u0 
        v = self.focal * (yc / zc) + self.v0
        
        if 0 <= u < self.width and 0 <= v < self.height:
            return (int(u), int(v))
        return None
    
    
    
    # ------------- Obtenir les coordonnées normalisées de la cible + l'interface de la camera
    def get_view(self, robot_pose, targets, obstacles_list, s_star=None): 
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Environnement camera (Ciel et Sol)
        cv2.rectangle(img, (0, 0), (self.width, self.height//2), (150, 150, 200), -1)
        cv2.rectangle(img, (0, self.height//2), (self.width, self.height), (100, 100, 50), -1)
        
        # Vue des points des cibles s et s* dans la camera
        cibles_visibles = []
        for i, group_pts_cible in enumerate(targets): # Cible s
            pixels = []
            for j, point in enumerate(group_pts_cible):
                pixel = self.changement_repere(point, robot_pose)
                if pixel:
                    u_int, v_int = int(pixel[0]), int(pixel[1])
                    pixels.append((u_int, v_int))
                    # Dessin cible s (les points)
                    cv2.circle(img,pixel, 4, (0, 255, 255), -1) # points courants en jaune
                    # Numérotation pour vérification de l'ordre des points dans l'image
                    cv2.putText(img, str(j), (u_int+5, v_int-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                # Trajectoire dans l'espace image
                if s_star is not None: # Cible s*
                    u_s = int(s_star[2*j])
                    v_s = int(s_star[2*j+1])
                    cv2.line(img, pixel, (u_s, v_s), (0, 255, 0), 1) # Ligne de rappel (Verte)  
                    cv2.circle(img, (u_s, v_s), 4, (0, 0, 255), 1) # Dessin Cible s* (Points Rouge)
                    
        
        # ------------------------------------------ Dessin Cible rectangle plein --------------------------------------------------  
            if len(pixels) == 4:
                cibles_visibles.append((i, pixels))
                # Dessin du contour du panneau pour la forme
                pts = np.array(pixels, np.int32).reshape((-1, 1, 2))
                # cv2.polylines(img, [pts], True, (255, 255, 255), 1) # rectangle creux
                cv2.fillPoly(img, [pts], (0, 200, 180)) # Rectangle plein
        
        
        ##------------------------------------------ Dessin Cible Qrcode -----------------------------------------------------------
        #     if len(pixels) == 4:
        #         cibles_visibles.append((i, pixels))
                
        #         qr_img = cv2.imread('simulateur_python_IBVS/Test_IBVS_3DD_ameliorée/mon_qrcode.png')
                
        #         if qr_img is not None:
        #             h_q, w_q = qr_img.shape[:2]
        #             # On force le type float32 pour les points
        #             src_pts = np.array([[0,0], [w_q,0], [w_q,h_q], [0,h_q]], dtype=np.float32)
        #             dst_pts = np.array(pixels, dtype=np.float32)
                    
        #             # findHomography renvoie (matrice, masque), on ne veut que la matrice [0]
        #             h_mat, _ = cv2.findHomography(src_pts, dst_pts)
                    
        #             # On vérifie que h_mat n'est pas vide avant de déformer
        #             if h_mat is not None:
        #                 # On applique la transformation
        #                 img = cv2.warpPerspective(qr_img, h_mat, (self.width, self.height),dst=img, borderMode=cv2.BORDER_TRANSPARENT)

        # # ------------------------------------------------------------------------------------------------------------------------------
        
        # --------- Dessin du centre optique (croix verte)
        cv2.drawMarker(img, (int(self.u0), int(self.v0)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # --------- Dessin des obstacles
        for ox, oy, r in obstacles_list:
            p_bas = self.changement_repere((ox, oy, -0.4), robot_pose)
            p_haut = self.changement_repere((ox, oy, 0.4), robot_pose)
            if p_bas and p_haut:
                cv2.line(img, p_bas, p_haut, (0, 0, 250), 10)
        
        return img, cibles_visibles