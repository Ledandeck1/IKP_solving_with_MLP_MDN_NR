import numpy as np


DH_TABLE = np.array([
    [6.5,  0.0,   np.pi/2],   # Joint 1 (Base)
    [0.0,  11.0,  0.0],       # Joint 2 (Shoulder)
    [0.0,  15.0,  0.0],       # Joint 3 (Elbow)
    [0.0,  0.0,   np.pi/2],   # Joint 4 (Wrist Pitch)
    [0.0,  0.0,  -np.pi/2],   # Joint 5 (Wrist Roll)
    [18.0, 0.0,   0.0]        # Joint 6 (End-Effector)
])

def get_dh_matrix(theta, d, a, alpha):
    """Calcula a matrix 4x4 de transformação do D-H."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,    0.0,  1.0]
    ])

def ForwardKinematic(theta0, theta1, theta2, theta3, theta4, theta5):
    """
    Retorna apenas a posicao do EE em array de [X, Y, Z]

    """
    thetas = [theta0, theta1, theta2, theta3, theta4, theta5]
    T = np.eye(4)
    
    for i in range(6):
        A = get_dh_matrix(thetas[i], DH_TABLE[i,0], DH_TABLE[i,1], DH_TABLE[i,2])
        T = np.dot(T, A)
        
    return T[:3, 3]


