import numpy as np
import casadi as ca

M_CART = 2.0
M_POLE = 1.0
M_TOTAL = M_CART + M_POLE
L_POLE = 1.0
MPLP = M_POLE*L_POLE
G = 9.81
MPG = M_POLE*G
MTG = M_TOTAL*G
MTLP = M_TOTAL*G
PI_2 = 2*np.pi
PI_UNDER_2 = 2/np.pi
PI_UNDER_1 = 1/np.pi
TS = 0.01

NUM_STATE = 5
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
R = 0.001
P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])
HOR = 64

def EulerForwardCartpole_virtual_Casadi(dynamic_update_virtual_Casadi, dt, x,u) -> ca.vertcat:
    xdot = dynamic_update_virtual_Casadi(x,u)
    return x + xdot * dt

def dynamic_update_virtual_Casadi(x, u) -> ca.vertcat:
    # Return the derivative of the state
    # u is 1x1 array, covert to scalar by u[0] 
        
    return ca.vertcat(
        x[1],            # xdot 
        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u[0] 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        x[3],        # thetadot
        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u[0]
          )/(MTLP - MPLP*np.cos(x[2])**2),  # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    )
    
def ThetaToRedTheta(theta):
    return (theta-np.pi)**2/-np.pi + np.pi

def MPC_Solve( system_update, system_dynamic, x0:np.array, initial_guess_x:float, initial_guess_u:float, num_state:int, horizon:int, Q_cost:np.array, R_cost:float, P_cost:np.array, ts: float, opts_setting ):
    retries = 0
    max_retries = 3
    while retries<max_retries:
        try:
            # casadi_Opti
            optimizer_normal = ca.Opti()
            
            ##### normal mpc #####  
            # x and u mpc prediction along N
            X_pre = optimizer_normal.variable(num_state, horizon + 1) 
            U_pre = optimizer_normal.variable(1, horizon)
            # set intial guess
            optimizer_normal.set_initial(X_pre, initial_guess_x)
            optimizer_normal.set_initial(U_pre, initial_guess_u)

            optimizer_normal.subject_to(X_pre[:, 0] == x0)  # starting state

            # cost 
            cost = 0

            # initial cost
            cost += Q_cost[0,0]*X_pre[0, 0]**2 + Q_cost[1,1]*X_pre[1, 0]**2 + Q_cost[2,2]*X_pre[2, 0]**2 + Q_cost[3,3]*X_pre[3, 0]**2 + Q_cost[4,4]*X_pre[4, 0]**2

            # state cost
            for k in range(0,horizon-1):
                x_next = system_update(system_dynamic,ts,X_pre[:, k],U_pre[:, k])
                optimizer_normal.subject_to(X_pre[:, k + 1] == x_next)
                cost += Q_cost[0,0]*X_pre[0, k+1]**2 + Q_cost[1,1]*X_pre[1, k+1]**2 + Q_cost[2,2]*X_pre[2, k+1]**2 + Q_cost[3,3]*X_pre[3, k+1]**2 + Q_cost[4,4]*X_pre[4, k+1]**2 + R_cost * U_pre[:, k]**2

            # terminal cost
            x_terminal = system_update(system_dynamic,ts,X_pre[:, horizon-1],U_pre[:, horizon-1])
            optimizer_normal.subject_to(X_pre[:, horizon] == x_terminal)
            cost += P_cost[0,0]*X_pre[0, horizon]**2 + P_cost[1,1]*X_pre[1, horizon]**2 + P_cost[2,2]*X_pre[2, horizon]**2 + P_cost[3,3]*X_pre[3, horizon]**2 + P_cost[4,4]*X_pre[4, horizon]**2 + R_cost * U_pre[:, horizon-1]**2

            optimizer_normal.minimize(cost)
            optimizer_normal.solver('ipopt',opts_setting)
            sol = optimizer_normal.solve()
            X_sol = sol.value(X_pre)
            U_sol = sol.value(U_pre)
            Cost_sol = sol.value(cost)
            break
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            if x0[2] >= np.pi: 
                initial_guess_x = -5
                initial_guess_u = -1000
            else:
                initial_guess_x = 5
                initial_guess_u = 1000
            retries += 1
    return X_sol, U_sol, Cost_sol


def main():
    x_0 = 0
    theta = np.pi
    theta_red = ThetaToRedTheta(theta)
    x0 = np.array([x_0,0,theta,0,theta_red])
    x0 = np.array([0.5       ,  0.        ,  2.96034692,  0.        ,  3.13113617])
    x_ini_guess = -5
    u_ini_guess = -2000000
    opts_setting = {'ipopt.max_iter':5000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, P, TS, opts_setting)
    print("X_sol[1]=",X_sol[:,1])
    print("U_sol[0]=",U_sol[0])

if __name__ == "__main__":
    main()