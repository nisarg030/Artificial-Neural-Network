L = input('Enter the no. of neurons in input layer: ');
N = input('Enter the no. of neurons in output layer: ');
M = input('Enter the no. of neurons in hidden layer: ');
C = input('Enter the no. of paterns that we have to train: ');
D = input('Enter the no. of paterns that we have to test: ');
matrix = readmatrix('Book1.csv');
I = matrix(1:L,1:C);
T = matrix(L+1:end,1:C);
P=size(I,2);% no. of patterns

a_1 = 1;% constant
a_2 = 1;% constant
eta = 0.5;% learning rate
alpha = 0.3;% momentum coefficient
MSE = 1; % initial guess to start loop 
error = 0.01; % MSE limit
ite = 0;
%for V
for i=1:(L+1)
    for j=1:M
        V(i,j) = -1 + 2.*rand();
    end
end

%for W
for j=1:(M+1) 
    for k=1:N
        W(j,k) = -1 + 2.*rand();
    end
end

% normalizing data for input matrix
for i = 1:L
    I_maximum = max(I(i,:));
    I_minimum = min(I(i,:));
    for p = 1:P
        I(i,p) = 0.1 + 0.8 * (I(i,p) - I_minimum) / (I_maximum - I_minimum);
    end
end

% normalizing data for target matrix
for i = 1:N
    T_maximum = max(T(i,:));
    T_minimum = min(T(i,:));
    for p = 1:P
        T(i,p) = 0.1 + 0.8 * (T(i,p) - T_minimum) / (T_maximum - T_minimum);
    end
end

% values of change in w and v for previous iteration
w_delta = zeros(M+1,N); 
v_delta = zeros(L+1,M);


while MSE > error
% calculation of output of hidden neuron
    
I_h = zeros(M,P); 
O_h = zeros(M,P); 
for p=1:P
    for j=1:M
        for i=2:(L+1)
            I_h(j,p) = I_h(j,p) + I(i-1,p).*V(i,j);
        end
        I_h(j,p) = I_h(j,p) + 1*V(1,j);
        O_h(j,p) = 1/(1+exp((-a_1).*I_h(j,p)));
    end
end
%% calculation of output of output neurons
I_o = zeros(N,P); 
O_o = zeros(N,P); 
E = zeros(N,P); 
MSE = 0; 
MM= 0;
for p=1:P
    for k=1:N
        for j=2:(M+1)
            I_o(k,p) = I_o(k,p) + O_h(j-1,p)*W(j,k);
        end
        I_o(k,p) = I_o(k,p) + 1*W(1,k);
        O_o(k,p) = (exp((a_2).*I_o(k,p))-exp((-a_2).*I_o(k,p)))/(exp((a_2).*I_o(k,p))+exp((-a_2).*I_o(k,p)));
        E(k,p) = (1/2) * (T(k,p) - O_o(k,p))^2;
    end
     MSE = MSE + sum(E(:,p));
end
 MSE = MSE/P
%% Updatation of V and W

% Updatation of W
for j = 2:(M+1)
    for k = 1:N
        w_jk = 0;
        for p = 1:P
            w_jk = w_jk + ((T(k,p) - O_o(k,p)) * a_2 * (1 - (O_o(k,p))^2) * O_h(j-1,p));
        end
        W(j,k) = W(j,k) + (eta / P) * w_jk + alpha * w_delta(j,k);
        w_delta(j,k) = (eta / P) * w_jk + alpha * w_delta(j,k);
    end
end
% to update bias weight w1
for k = 1:N
    w_1k = 0;
    for p = 1:P
         w_1k = w_1k + ((T(k,p) - O_o(k,p)) * a_2 * (1 - (O_o(k,p))^2) * 1);
    end
    W(1,k) = W(1,k) + (eta / P) * w_1k + alpha * w_delta(1,k);
    w_delta(1,k) = (eta / P) * w_1k + alpha * w_delta(1,k);
end
% Updatation of V
for i = 2:(L+1)
    for j = 1:M
        v_ij = 0;
        for p = 1:P
            v = 0;
            for k = 1:N
                v = v + ((T(k,p) - O_o(k,p)) * a_2 * (1 - (O_o(k,p))^2) * W(j+1,k) * a_1 * O_h(j,p) * (1 - O_h(j,p)) * I(i-1,p));
            end
            v_ij = v_ij + v;
        end
        V(i,j) = V(i,j) + (eta / (N*P)) * v_ij + alpha * v_delta(i,j);
        v_delta(i,j) = (eta / (N*P)) * v_ij + alpha * v_delta(i,j);
    end
end
% to update bias weight v1
for j = 1:M
    v_1j = 0;
    for p = 1:P
        v = 0;
        for k = 1:N
             v = v + ((T(k,p) - O_o(k,p)) * a_2 * (1 - (O_o(k,p))^2) * W(j+1,k) * a_1 * O_h(j,p) * (1 - O_h(j,p)) * 1);
        end
         v_1j = v_1j + v;
    end
     V(1,j) = V(1,j) + (eta / (N*P)) * v_1j + alpha * v_delta(1,j);
     v_delta(1,j) = (eta / (N*P)) * v_1j + alpha * v_delta(1,j);
end
ite = ite + 1
MSE_matrix(ite) = MM + MSE;
end
writematrix(MSE_matrix,'outputmse.csv')
disp(W)
disp(V)

%testing of nural network
I_test = matrix(1:L,C+1:C+D);
T_test = matrix(L+1:end,C+1:C+D);
P_test=size(I_test,2);

for i = 1:L
    I_maximum = max(I_test(i,:));
    I_minimum = min(I_test(i,:));
    for p = 1:P_test
        I_test(i,p) = 0.1 + 0.8 * (I_test(i,p) - I_minimum) / (I_maximum - I_minimum);
    end
end

for i = 1:N
    T_maximum = max(T_test(i,:));
    T_minimum = min(T_test(i,:));
    for p = 1:P_test
        T_test(i,p) = 0.1 + 0.8 * (T_test(i,p) - T_minimum) / (T_maximum - T_minimum);
    end
end
I_h = zeros(M,P_test); 
O_h = zeros(M,P_test); 
for p=1:P_test
    for j=1:M
        for i=2:(L+1)
            I_h(j,p) = I_h(j,p) + I_test(i-1,p)*V(i,j);
        end
        I_h(j,p) = I_h(j,p) + 1*V(1,j);
        O_h(j,p) = 1/(1+exp((-a_1).*I_h(j,p)));
    end
end


I_o = zeros(N,P_test); % Input to hidden layer
O_o = zeros(N,P_test); 
E = zeros(N,P_test); % Error
MSE_test = 0; % Mean Square Error
error_test = zeros(N,P_test);

for p=1:P_test
    for k=1:N
        for j=2:(M+1)
            I_o(k,p) = I_o(k,p) + O_h(j-1,p)*W(j,k);
        end
        I_o(k,p) = I_o(k,p) + 1*W(1,k);
        O_o(k,p) = (exp((a_2).*I_o(k,p))-exp((-a_2).*I_o(k,p)))/(exp((a_2).*I_o(k,p))+exp((-a_2).*I_o(k,p)));
        E(k,p) = (1/2) * (T_test(k,p) - O_o(k,p))^2;
        error_test(k,p) = T_test(k,p) - O_o(k,p);
        writematrix(error_test,'outputet.csv')
    end
     MSE_test = MSE_test + sum(E(:,p));
end
MSE_test = MSE_test/P_test
for i = 1:N
    O_maximum = max(O_o(i,:));
    O_minimum = min(O_o(i,:));
    for p = 1:P_test
        O_o(i,p) = O_minimum + (O_o(i,p) - 0.1) * (T_maximum - T_minimum)/(0.8);
    end
end
 writematrix(O_o,'outputnetwork.csv')

