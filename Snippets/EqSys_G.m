syms N_1 N_2 N_3 N_4 N_5 f_1 f_2 f_3 f_4

eqn1 = N_1 + f_1- 1 == N_2;
eqn2 = N_2 + f_2 - 1 == N_3;
eqn3 = N_3 + f_3 - 1 == N_4;
eqn4 = N_4 + f_4 - 1 == N_5;
eqn5 = N_5 == 231;
eqn6 = f_1 == 5;
eqn7 = f_2 == 2*f_1; 
eqn8 = f_3 == 2*f_2;
eqn9 = f_4 == 2*f_3;

sol = solve([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9], [N_1, N_2, N_3, N_4, N_5, f_1, f_2, f_3, f_4]);
N_1_sol = sol.N_1
N_2_sol = sol.N_2
N_3_sol = sol.N_3
N_4_sol = sol.N_4
N_5_sol = sol.N_5
f_1_sol = sol.f_1
f_2_sol = sol.f_2
f_3_sol = sol.f_3
f_4_sol = sol.f_4