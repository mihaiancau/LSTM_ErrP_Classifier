syms N_1 N_2 N_3 N_4 N_5 f_1 f_2 f_3 f_4 f_5 p_1 p_2 p_3 p_4 s_1 s_2 s_3 s_4

eqn1 = (N_2 - 1)*s_1 + f_1 - 2*p_1 == N_1;
eqn2 = (N_3 - 1)*s_2 + f_2 - 2*p_2 == N_2;
eqn3 = (N_4 - 1)*s_3 + f_3 - 2*p_3 == N_3;
eqn4 = (N_5 - 1)*s_4 + f_4 - 2*p_4 == N_4;
eqn5 = f_5 == N_5;
eqn6 = f_4 == 3; 
eqn7 = f_3 == 7;
eqn8 = f_2 == 16;
eqn9 = f_1 == 33;
eqn10 = s_1 == 2;
eqn11 = s_2 == 2;
eqn12 = s_3 == 2;
eqn13 = s_4 == 2;
eqn14 = p_1 == 1;
eqn15 = p_2 == 1;
eqn16 = p_3 == 1;
eqn17 = p_4 == 1;
eqn18 = N_1 == 231;

sol = solve([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9, eqn10, eqn11, eqn12, eqn13, eqn14, eqn15, eqn16, eqn17, eqn18], [N_1, N_2, N_3, N_4, N_5, f_1, f_2, f_3, f_4, f_5, p_1, p_2, p_3, p_4, s_1, s_2, s_3, s_4]);
f_5_sol = sol.f_5