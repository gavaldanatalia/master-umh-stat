% Ejercico 1
A = [1/sqrt(3) 1/sqrt(3) 1/sqrt(3); 1/sqrt(2) -1/sqrt(2) 0; 1/sqrt(6) 1/sqrt(6) -2/sqrt(6)];
A_inv = inv(A);
det(A);

a_por_ainv = A * A_inv;

B = [1/sqrt(2) 0 -1/sqrt(2); 1/sqrt(2) 0 1/sqrt(2); 0 1 0];
B_inv = inv(B);
det(B);

C = A*B;
C_inv = inv(C);
C_t = transpose(C);

c_por_cinv = C * C_inv;

c_por_t = C * C_t;

% Ejercico 4
global a b c;
A = [0 a -b; -a 0 c; b -c 0]